#include <immintrin.h>
#include <inttypes.h>

#include "mlasi.h"

template <typename DataType>
static __m256i
Pack16Bits(__m256i a, __m256i b);

template <>
__m256i
Pack16Bits<uint8_t>(__m256i a, __m256i b)
{
    return _mm256_packus_epi16(a, b);
}

template <>
__m256i
Pack16Bits<int8_t>(__m256i a, __m256i b)
{
    return _mm256_packs_epi16(a, b);
}


////////////////////////////////////////////////////////////////////////////////
// Strict XNNPack method
////////////////////////////////////////////////////////////////////////////////
#include <assert.h>

static uint32_t BitsOfFp32(float f) {
    union {
        uint32_t u32;
        float    fp32;
    } uf;
    uf.fp32 = f;
    return uf.u32;
}

static float Fp32FromBits(uint32_t u) {
    union {
        uint32_t u32;
        float    fp32;
    } uf = {u};
    return uf.fp32;
}

template <typename DataType>
static __m256i
ShiftRight8Epi16(__m256i v);

template <>
__m256i
ShiftRight8Epi16<int8_t>(__m256i v)
{
    return _mm256_srai_epi16(v, 8);
}

template <>
__m256i
ShiftRight8Epi16<uint8_t>(__m256i v)
{
    return _mm256_srli_epi16(v, 8);
}


template<typename DataType, bool IsScalarA, bool IsScalarB>
void
MlasQLinearAddKernelAvx2Helper_XNNPack(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t p_N
    )
{
    float ScaleRatio_AC = ScaleA / ScaleC;
    float ScaleRatio_BC = ScaleB / ScaleC;

    constexpr float MinScaleRatio = 6.103515625e-05f; // std::stof("0x1.0p-14f");
    constexpr float MaxScaleRatio = 256.0f; //std::stof("0x1.0p+8f");
    (void)MinScaleRatio;
    (void)MaxScaleRatio;
    assert(ScaleRatio_AC >= MinScaleRatio && ScaleRatio_AC < MaxScaleRatio);
    assert(ScaleRatio_BC >= MinScaleRatio && ScaleRatio_BC < MaxScaleRatio);

    const float GreaterScaleRatio = std::max(ScaleRatio_AC, ScaleRatio_BC);
    const int32_t GreaterExponent = (int32_t)(BitsOfFp32(GreaterScaleRatio) >> 23) - 127;
    const uint32_t Shift = (uint32_t)(21 - GreaterExponent); // Shift is in [13, 31] range.
    assert(Shift <= 31 && Shift >= 13);

    // Get 22 effective bits for the bigger scale ratio (float in fact has 23 + 1 effective bits)
    const float MultiplierForMantissa = Fp32FromBits((uint32_t)(21 - GreaterExponent + 127) << 23);
    const int32_t MultiplierA = (int32_t) lrintf(ScaleRatio_AC * MultiplierForMantissa);
    const int32_t MultiplierB = (int32_t) lrintf(ScaleRatio_BC * MultiplierForMantissa);
    assert(MultiplierA < 0x00400000 && MultiplierB < 0x00400000);
    assert(MultiplierA >= 0x00200000 || MultiplierB >= 0x00200000); // the bigger one must fullfil this check

    const __m256i Int16VectorMultiplierA_lo = _mm256_set1_epi16(MultiplierA & 0xFFFF);
    const __m256i Int16VectorMultiplierA_hi = _mm256_set1_epi16(MultiplierA >> 16);
    const __m256i Int16VectorMultiplierB_lo = _mm256_set1_epi16(MultiplierB & 0xFFFF);
    const __m256i Int16VectorMultiplierB_hi = _mm256_set1_epi16(MultiplierB >> 16);
    const int32_t ZeroPointConstPart = -(MultiplierA * ZeroPointA + MultiplierB * ZeroPointB);
    __m256i VectorZeroPointConstPart = _mm256_set1_epi32(ZeroPointConstPart);
    const __m128i vshift = _mm_cvtsi32_si128((int)Shift);

    const int32_t RemainderMask = (1 << Shift) - 1;
    const int32_t RemainderThreshold = RemainderMask >> 1;
    const __m256i vremainder_mask = _mm256_set1_epi32(RemainderMask);
    const __m256i vremainder_threshold = _mm256_set1_epi32(RemainderThreshold);

    if (IsScalarA) {
        const auto a_32xi8 = _mm256_set1_epi8(*(const char*)InputA);
        const auto a_16xi16 = ShiftRight8Epi16<DataType>(_mm256_unpacklo_epi8(a_32xi8, a_32xi8));
        const auto ap_lo = _mm256_mullo_epi16(a_16xi16, Int16VectorMultiplierA_lo);
        const auto ap_hi = _mm256_add_epi16(_mm256_mulhi_epu16(a_16xi16, Int16VectorMultiplierA_lo), _mm256_mullo_epi16(a_16xi16, Int16VectorMultiplierA_hi));
        VectorZeroPointConstPart = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpacklo_epi16(ap_lo, ap_hi));
    }
    if (IsScalarB) {
        const auto b_32xi8 = _mm256_set1_epi8(*(const char*)InputB);
        const auto b16x16 = ShiftRight8Epi16<DataType>(_mm256_unpacklo_epi8(b_32xi8, b_32xi8));
        const auto bp_lo = _mm256_mullo_epi16(b16x16, Int16VectorMultiplierB_lo);
        const auto bp_hi = _mm256_add_epi16(_mm256_mulhi_epu16(b16x16, Int16VectorMultiplierB_lo), _mm256_mullo_epi16(b16x16, Int16VectorMultiplierB_hi));
        VectorZeroPointConstPart = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpacklo_epi16(bp_lo, bp_hi));
    }

    int64_t N = static_cast<int64_t>(p_N);
    __m256i vy = _mm256_setzero_si256();
    while (N > 0) {
        __m256i a_32xi8, b_32xi8, vy02, vy13;

        // First half (16 x values)
        {
            __m256i ap_lo, ap_hi;
            __m256i bp_lo, bp_hi;
            if (!IsScalarA) {
                a_32xi8 = _mm256_lddqu_si256((const __m256i*)InputA);
                InputA += 32;
                const auto a_16xi16 = ShiftRight8Epi16<DataType>(_mm256_unpacklo_epi8(a_32xi8, a_32xi8));
                ap_lo = _mm256_mullo_epi16(a_16xi16, Int16VectorMultiplierA_lo);
                ap_hi = _mm256_add_epi16(_mm256_mulhi_epu16(a_16xi16, Int16VectorMultiplierA_lo), _mm256_mullo_epi16(a_16xi16, Int16VectorMultiplierA_hi));
            }
            if (!IsScalarB) {
                b_32xi8 = _mm256_lddqu_si256((const __m256i*)InputB);
                InputB += 32;
                const auto b16x16 = ShiftRight8Epi16<DataType>(_mm256_unpacklo_epi8(b_32xi8, b_32xi8));
                bp_lo = _mm256_mullo_epi16(b16x16, Int16VectorMultiplierB_lo);
                bp_hi = _mm256_add_epi16(_mm256_mulhi_epu16(b16x16, Int16VectorMultiplierB_lo), _mm256_mullo_epi16(b16x16, Int16VectorMultiplierB_hi));
            }

            __m256i vacc_lo, vacc_hi;
            if (IsScalarA) {
                vacc_lo = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpacklo_epi16(bp_lo, bp_hi));
                vacc_hi = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpackhi_epi16(bp_lo, bp_hi));
            } else if (IsScalarB) {
                vacc_lo = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpacklo_epi16(ap_lo, ap_hi));
                vacc_hi = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpackhi_epi16(ap_lo, ap_hi));
            } else {
                // Accumulate products.
                vacc_lo = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpacklo_epi16(ap_lo, ap_hi));
                vacc_hi = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpackhi_epi16(ap_lo, ap_hi));
                vacc_lo = _mm256_add_epi32(vacc_lo, _mm256_unpacklo_epi16(bp_lo, bp_hi));
                vacc_hi = _mm256_add_epi32(vacc_hi, _mm256_unpackhi_epi16(bp_lo, bp_hi));
            }
                
            //Shift right and round.
            const __m256i vrem_lo = _mm256_add_epi32(_mm256_and_si256(vacc_lo, vremainder_mask), _mm256_cmpgt_epi32(_mm256_setzero_si256(), vacc_lo));
            const __m256i vrem_hi = _mm256_add_epi32(_mm256_and_si256(vacc_hi, vremainder_mask), _mm256_cmpgt_epi32(_mm256_setzero_si256(), vacc_hi));
            vacc_lo = _mm256_sub_epi32(_mm256_sra_epi32(vacc_lo, vshift), _mm256_cmpgt_epi32(vrem_lo, vremainder_threshold));
            vacc_hi = _mm256_sub_epi32(_mm256_sra_epi32(vacc_hi, vshift), _mm256_cmpgt_epi32(vrem_hi, vremainder_threshold));

            // Pack, saturate, and add output zero point.
            const __m256i vy_zero_point = _mm256_set1_epi16((short)ZeroPointC);
            vy02 = _mm256_adds_epi16(_mm256_packs_epi32(vacc_lo, vacc_hi), vy_zero_point);
        }


        // Another half (16 x value)
        {
            __m256i ap_lo, ap_hi;
            __m256i bp_lo, bp_hi;
            if (!IsScalarA) {
                const auto a_16xi16 = ShiftRight8Epi16<DataType>(_mm256_unpackhi_epi8(a_32xi8, a_32xi8));
                ap_lo = _mm256_mullo_epi16(a_16xi16, Int16VectorMultiplierA_lo);
                ap_hi = _mm256_add_epi16(_mm256_mulhi_epu16(a_16xi16, Int16VectorMultiplierA_lo), _mm256_mullo_epi16(a_16xi16, Int16VectorMultiplierA_hi));
            }
            if (!IsScalarB) {
                const auto b16x16 = ShiftRight8Epi16<DataType>(_mm256_unpackhi_epi8(b_32xi8, b_32xi8));
                bp_lo = _mm256_mullo_epi16(b16x16, Int16VectorMultiplierB_lo);
                bp_hi = _mm256_add_epi16(_mm256_mulhi_epu16(b16x16, Int16VectorMultiplierB_lo), _mm256_mullo_epi16(b16x16, Int16VectorMultiplierB_hi));
            }

            __m256i vacc_lo, vacc_hi;
            if (IsScalarA) {
                vacc_lo = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpacklo_epi16(bp_lo, bp_hi));
                vacc_hi = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpackhi_epi16(bp_lo, bp_hi));
            } else if (IsScalarB) {
                vacc_lo = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpacklo_epi16(ap_lo, ap_hi));
                vacc_hi = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpackhi_epi16(ap_lo, ap_hi));
            } else {
                // Accumulate products.
                vacc_lo = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpacklo_epi16(ap_lo, ap_hi));
                vacc_hi = _mm256_add_epi32(VectorZeroPointConstPart, _mm256_unpackhi_epi16(ap_lo, ap_hi));
                vacc_lo = _mm256_add_epi32(vacc_lo, _mm256_unpacklo_epi16(bp_lo, bp_hi));
                vacc_hi = _mm256_add_epi32(vacc_hi, _mm256_unpackhi_epi16(bp_lo, bp_hi));
            }
                
            //Shift right and round.
            const __m256i vrem_lo = _mm256_add_epi32(_mm256_and_si256(vacc_lo, vremainder_mask), _mm256_cmpgt_epi32(_mm256_setzero_si256(), vacc_lo));
            const __m256i vrem_hi = _mm256_add_epi32(_mm256_and_si256(vacc_hi, vremainder_mask), _mm256_cmpgt_epi32(_mm256_setzero_si256(), vacc_hi));
            vacc_lo = _mm256_sub_epi32(_mm256_sra_epi32(vacc_lo, vshift), _mm256_cmpgt_epi32(vrem_lo, vremainder_threshold));
            vacc_hi = _mm256_sub_epi32(_mm256_sra_epi32(vacc_hi, vshift), _mm256_cmpgt_epi32(vrem_hi, vremainder_threshold));

            // Pack, saturate, and add output zero point.
            const __m256i vy_zero_point = _mm256_set1_epi16((short)ZeroPointC);
            vy13 = _mm256_adds_epi16(_mm256_packs_epi32(vacc_lo, vacc_hi), vy_zero_point);
        }

        vy = Pack16Bits<DataType>(vy02, vy13);

        N -= 32;
        if (N < 0) break;

        _mm256_storeu_si256((__m256i*)OutputC, vy);
        OutputC += 32;
    }

    if (N < 0) {
        N += 32;
        int k = static_cast<int>(N / 4);
        if (k > 0) {
            const __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(k), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            _mm256_maskstore_epi32((int*)OutputC, mask, vy);
            OutputC += k * 4;
        }

        int r = static_cast<int>(N % 4);
        if (r > 0) {
            auto permuted = _mm256_permutevar8x32_epi32(vy, _mm256_set1_epi32(k));
            uint32_t PackedValueC = (uint32_t)_mm256_extract_epi32(permuted, 0);
            for (int n = 0; n < r; ++n) {
                *((uint8_t*)OutputC + n) = (uint8_t)PackedValueC;
                PackedValueC >>= 8;
            }
        }
    }
}

//////////////////////////////////////////////////////////////////////////////////
// XNNPack but fp32 multiply
/////////////////////////////////////////////////////////////////////////////////
template <typename DataType>
static __m256i
ShiftRight24Epi32(__m256i v);

template <>
__m256i
ShiftRight24Epi32<int8_t>(__m256i v)
{
    return _mm256_srai_epi32(v, 24);
}

template <>
__m256i
ShiftRight24Epi32<uint8_t>(__m256i v)
{
    return _mm256_srli_epi32(v, 24);
}

// Direct use fp32 multiply
template<typename DataType, bool IsScalarA, bool IsScalarB>
void
MlasQLinearAddKernelAvx2Helper_XNNFloatMultiply(
    const DataType* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const DataType* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    DataType* OutputC,
    size_t p_N
    )
{
    float ScaleRatio_AC = ScaleA / ScaleC;
    float ScaleRatio_BC = ScaleB / ScaleC;
    const __m256 VectorScaleRatio_AC = _mm256_set1_ps(ScaleRatio_AC);
    const __m256 VectorScaleRatio_BC = _mm256_set1_ps(ScaleRatio_BC);

    const float ZeroPointConstPart = (float)ZeroPointC - (ScaleRatio_AC * ZeroPointA + ScaleRatio_BC * ZeroPointB);
    __m256 VectorZeroPointConstPart = _mm256_set1_ps(ZeroPointConstPart);

    if (IsScalarA) {
        const auto a_8xfp32 = _mm256_set1_ps((float)(int32_t)*InputA);
        VectorZeroPointConstPart = _mm256_add_ps(VectorZeroPointConstPart, _mm256_mul_ps(a_8xfp32, VectorScaleRatio_AC));
    }
    if (IsScalarB) {
        const auto b32x8 = _mm256_set1_ps((float)(int32_t)*InputB);
        VectorZeroPointConstPart = _mm256_add_ps(VectorZeroPointConstPart, _mm256_mul_ps(b32x8, VectorScaleRatio_BC));
    }

    int64_t N = static_cast<int64_t>(p_N);
    __m256i vy = _mm256_setzero_si256();
    while (N > 0) {
        __m256i a_32xi8, b_32xi8, vy02, vy13;

        // First half (16 x i8)
        {
            if (!IsScalarA) {
                a_32xi8 = _mm256_lddqu_si256((const __m256i*)InputA);
                InputA += 32;
            }
            if (!IsScalarB) {
                b_32xi8 = _mm256_lddqu_si256((const __m256i*)InputB);
                InputB += 32;
            }

            __m256 lolo_8xfp32, lohi_8xfp32;
            if (IsScalarA) {
                const auto b_16xi16 = _mm256_unpacklo_epi8(b_32xi8, b_32xi8);
                lolo_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(b_16xi16, b_16xi16)));
                lohi_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(b_16xi16, b_16xi16)));
                lolo_8xfp32 = _mm256_fmadd_ps(lolo_8xfp32, VectorScaleRatio_BC, VectorZeroPointConstPart);
                lohi_8xfp32 = _mm256_fmadd_ps(lohi_8xfp32, VectorScaleRatio_BC, VectorZeroPointConstPart);
            } else if (IsScalarB) {
                const auto a_16xi16 = _mm256_unpacklo_epi8(a_32xi8, a_32xi8);
                lolo_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(a_16xi16, a_16xi16)));
                lohi_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(a_16xi16, a_16xi16)));
                lolo_8xfp32 = _mm256_fmadd_ps(lolo_8xfp32, VectorScaleRatio_AC, VectorZeroPointConstPart);
                lohi_8xfp32 = _mm256_fmadd_ps(lohi_8xfp32, VectorScaleRatio_AC, VectorZeroPointConstPart);
            } else {
                const auto b_16xi16 = _mm256_unpacklo_epi8(b_32xi8, b_32xi8);
                lolo_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(b_16xi16, b_16xi16)));
                lohi_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(b_16xi16, b_16xi16)));
                lolo_8xfp32 = _mm256_fmadd_ps(lolo_8xfp32, VectorScaleRatio_BC, VectorZeroPointConstPart);
                lohi_8xfp32 = _mm256_fmadd_ps(lohi_8xfp32, VectorScaleRatio_BC, VectorZeroPointConstPart);

                const auto a_16xi16 = _mm256_unpacklo_epi8(a_32xi8, a_32xi8);
                const auto alolo_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(a_16xi16, a_16xi16)));
                const auto alohi_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(a_16xi16, a_16xi16)));
                lolo_8xfp32 = _mm256_fmadd_ps(alolo_8xfp32, VectorScaleRatio_AC, lolo_8xfp32);
                lohi_8xfp32 = _mm256_fmadd_ps(alohi_8xfp32, VectorScaleRatio_AC, lohi_8xfp32);
            }

            vy02 = _mm256_packs_epi32(_mm256_cvtps_epi32(lolo_8xfp32), _mm256_cvtps_epi32(lohi_8xfp32));
        }


        // Another half (16 x i8)
        {
            __m256 hilo_8xfp32, hihi_8xfp32;
            if (IsScalarA) {
                const auto b_16xi16 = _mm256_unpackhi_epi8(b_32xi8, b_32xi8);
                hilo_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(b_16xi16, b_16xi16)));
                hihi_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(b_16xi16, b_16xi16)));
                hilo_8xfp32 = _mm256_fmadd_ps(hilo_8xfp32, VectorScaleRatio_BC, VectorZeroPointConstPart);
                hihi_8xfp32 = _mm256_fmadd_ps(hihi_8xfp32, VectorScaleRatio_BC, VectorZeroPointConstPart);
            } else if (IsScalarB) {
                const auto a_16xi16 = _mm256_unpackhi_epi8(a_32xi8, a_32xi8);
                hilo_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(a_16xi16, a_16xi16)));
                hihi_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(a_16xi16, a_16xi16)));
                hilo_8xfp32 = _mm256_fmadd_ps(hilo_8xfp32, VectorScaleRatio_AC, VectorZeroPointConstPart);
                hihi_8xfp32 = _mm256_fmadd_ps(hihi_8xfp32, VectorScaleRatio_AC, VectorZeroPointConstPart);
            } else {
                const auto b_16xi16 = _mm256_unpackhi_epi8(b_32xi8, b_32xi8);
                hilo_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(b_16xi16, b_16xi16)));
                hihi_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(b_16xi16, b_16xi16)));
                hilo_8xfp32 = _mm256_fmadd_ps(hilo_8xfp32, VectorScaleRatio_BC, VectorZeroPointConstPart);
                hihi_8xfp32 = _mm256_fmadd_ps(hihi_8xfp32, VectorScaleRatio_BC, VectorZeroPointConstPart);

                const auto a_16xi16 = _mm256_unpackhi_epi8(a_32xi8, a_32xi8);
                const auto ahilo_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpacklo_epi16(a_16xi16, a_16xi16)));
                const auto ahihi_8xfp32 = _mm256_cvtepi32_ps(ShiftRight24Epi32<DataType>(_mm256_unpackhi_epi16(a_16xi16, a_16xi16)));
                hilo_8xfp32 = _mm256_fmadd_ps(ahilo_8xfp32, VectorScaleRatio_AC, hilo_8xfp32);
                hihi_8xfp32 = _mm256_fmadd_ps(ahihi_8xfp32, VectorScaleRatio_AC, hihi_8xfp32);
            }

            vy13 = _mm256_packs_epi32(_mm256_cvtps_epi32(hilo_8xfp32), _mm256_cvtps_epi32(hihi_8xfp32));
        }

        vy = Pack16Bits<DataType>(vy02, vy13);

        N -= 32;
        if (N < 0) break;

        _mm256_storeu_si256((__m256i*)OutputC, vy);
        OutputC += 32;
    }

    if (N < 0) {
        N += 32;
        int k = static_cast<int>(N / 4);
        if (k > 0) {
            const __m256i mask = _mm256_cmpgt_epi32(_mm256_set1_epi32(k), _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0));
            _mm256_maskstore_epi32((int*)OutputC, mask, vy);
            OutputC += k * 4;
        }

        int r = static_cast<int>(N % 4);
        if (r > 0) {
            auto permuted = _mm256_permutevar8x32_epi32(vy, _mm256_set1_epi32(k));
            uint32_t PackedValueC = (uint32_t)_mm256_extract_epi32(permuted, 0);
            for (int n = 0; n < r; ++n) {
                *((uint8_t*)OutputC + n) = (uint8_t)PackedValueC;
                PackedValueC >>= 8;
            }
        }
    }
}


//////////////////////////////////////////////////////////////////////////////////
// Simple but strict with unroll
/////////////////////////////////////////////////////////////////////////////////
// template<typename DataType, bool IsScalarA, bool IsScalarB>
// void
// MlasQLinearAddKernelAvx2Helper_StrictWithUnroll(
// void
// MlasQLinearAddKernelAvx2Helper_XNNFloatMultiply(
//     const DataType* InputA,
//     float ScaleA,
//     int32_t ZeroPointA,
//     const DataType* InputB,
//     float ScaleB,
//     int32_t ZeroPointB,
//     float ScaleC,
//     int32_t ZeroPointC,
//     DataType* OutputC,
//     size_t p_N
//     )
// {
// }


///////////////////////////////////////////////////////////////////////////////////
// Wrapper for different entry
///////////////////////////////////////////////////////////////////////////////////
#include <mutex>
#include <cstdlib>

typedef void (* PQLinearAddHelperS8)(
    const int8_t* InputA, float ScaleA, int32_t ZeroPointA,
    const int8_t* InputB, float ScaleB, int32_t ZeroPointB,
    float ScaleC, int32_t ZeroPointC, int8_t* OutputC, size_t Length);

void
MlasQLinearAddS8KernelAvx2(
    const int8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const int8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    int8_t* OutputC,
    size_t LengthA,
    size_t LengthB
) 
{
    static std::mutex mtx;
    static PQLinearAddHelperS8 helper_true_false = nullptr;
    static PQLinearAddHelperS8 helper_false_true = nullptr;
    static PQLinearAddHelperS8 helper_false_false = nullptr;

    if (helper_true_false == nullptr) {
        std::lock_guard<std::mutex> LockGuard(mtx);
        if (helper_true_false == nullptr) {
            const char* env_p = std::getenv("QLINEARADD_METHOD");
            if (env_p == nullptr || strcasecmp(env_p, "XNNFloatMultiply") == 0) {
                helper_true_false = MlasQLinearAddKernelAvx2Helper_XNNFloatMultiply<int8_t, true, false>;
                helper_false_true = MlasQLinearAddKernelAvx2Helper_XNNFloatMultiply<int8_t, false, true>;
                helper_false_false = MlasQLinearAddKernelAvx2Helper_XNNFloatMultiply<int8_t, false, false>;
            } else if (strcasecmp(env_p, "XNNPack") == 0){
                helper_true_false = MlasQLinearAddKernelAvx2Helper_XNNPack<int8_t, true, false>;
                helper_false_true = MlasQLinearAddKernelAvx2Helper_XNNPack<int8_t, false, true>;
                helper_false_false = MlasQLinearAddKernelAvx2Helper_XNNPack<int8_t, false, false>;
            } else {
                std::string msg("Wrong environment string on QLINEARADD_METHOD:");
                msg += env_p;
                throw std::runtime_error(msg);
            }
        }
    }

    if (LengthA == 1) {
        helper_true_false(
            InputA, ScaleA, ZeroPointA,
            InputB, ScaleB, ZeroPointB,
            ScaleC, ZeroPointC, OutputC,
            LengthB
        );
    } else if (LengthB == 1) {
        helper_false_true(
            InputA, ScaleA, ZeroPointA,
            InputB, ScaleB, ZeroPointB,
            ScaleC, ZeroPointC, OutputC,
            LengthA
        );
    } else {
        helper_false_false(
            InputA, ScaleA, ZeroPointA,
            InputB, ScaleB, ZeroPointB,
            ScaleC, ZeroPointC, OutputC,
            LengthA
        );
    }
}

typedef void (* PQLinearAddHelperU8)(
    const uint8_t* InputA, float ScaleA, int32_t ZeroPointA,
    const uint8_t* InputB, float ScaleB, int32_t ZeroPointB,
    float ScaleC, int32_t ZeroPointC, uint8_t* OutputC, size_t Length);

void
MlasQLinearAddU8KernelAvx2(
    const uint8_t* InputA,
    float ScaleA,
    int32_t ZeroPointA,
    const uint8_t* InputB,
    float ScaleB,
    int32_t ZeroPointB,
    float ScaleC,
    int32_t ZeroPointC,
    uint8_t* OutputC,
    size_t LengthA,
    size_t LengthB
    )
{
    static std::mutex mtx;
    static PQLinearAddHelperU8 helper_true_false = nullptr;
    static PQLinearAddHelperU8 helper_false_true = nullptr;
    static PQLinearAddHelperU8 helper_false_false = nullptr;

    if (helper_true_false == nullptr) {
        std::lock_guard<std::mutex> LockGuard(mtx);
        if (helper_true_false == nullptr) {
            const char* env_p = std::getenv("QLINEARADD_METHOD");
            if (env_p == nullptr || strcasecmp(env_p, "XNNFloatMultiply") == 0) {
                helper_true_false = MlasQLinearAddKernelAvx2Helper_XNNFloatMultiply<uint8_t, true, false>;
                helper_false_true = MlasQLinearAddKernelAvx2Helper_XNNFloatMultiply<uint8_t, false, true>;
                helper_false_false = MlasQLinearAddKernelAvx2Helper_XNNFloatMultiply<uint8_t, false, false>;
            } else if (strcasecmp(env_p, "XNNPack") == 0){
                helper_true_false = MlasQLinearAddKernelAvx2Helper_XNNPack<uint8_t, true, false>;
                helper_false_true = MlasQLinearAddKernelAvx2Helper_XNNPack<uint8_t, false, true>;
                helper_false_false = MlasQLinearAddKernelAvx2Helper_XNNPack<uint8_t, false, false>;
            } else {
                std::string msg("Wrong environment string on QLINEARADD_METHOD:");
                msg += env_p;
                throw std::runtime_error(msg);
            }
        }
    }

    if (LengthA == 1) {
        helper_true_false(
            InputA, ScaleA, ZeroPointA,
            InputB, ScaleB, ZeroPointB,
            ScaleC, ZeroPointC, OutputC,
            LengthB
        );
    } else if (LengthB == 1) {
        helper_false_true(
            InputA, ScaleA, ZeroPointA,
            InputB, ScaleB, ZeroPointB,
            ScaleC, ZeroPointC, OutputC,
            LengthA
        );
    } else {
        helper_false_false(
            InputA, ScaleA, ZeroPointA,
            InputB, ScaleB, ZeroPointB,
            ScaleC, ZeroPointC, OutputC,
            LengthA
        );
    }
}
