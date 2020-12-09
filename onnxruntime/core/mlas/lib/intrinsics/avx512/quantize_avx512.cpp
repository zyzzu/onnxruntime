/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    quantize.cpp

Abstract:

    This module implements routines to quantize buffers.

    For quantization formula as specified in the ONNX operator documentation is:

        Output = Saturate(RoundToEven(Input / Scale) + ZeroPoint)

--*/

#include "mlasi.h"

//
// QuantizeLinear implementation using AVX512 intrinsics.
//

MLAS_FORCEINLINE
__m512i
MlasQuantizeLinearVector(
    __m512 FloatVector,
    __m512 ScaleVector,
    __m512 MinimumValueVector,
    __m512 MaximumValueVector,
    __m512i ZeroPointVector) {
  //
  // Scale the input vector and clamp the values to the minimum and maximum
  // range (adjusted by the zero point value).
  //

  FloatVector = _mm512_div_ps(FloatVector, ScaleVector);
  // N.B. MINPS and MAXPS returns the value from the second vector if the
  // value from the first vector is a NaN.
  FloatVector = _mm512_max_ps(FloatVector, MinimumValueVector);
  FloatVector = _mm512_min_ps(FloatVector, MaximumValueVector);

  //
  // Convert the float values to integer using "round to nearest even" and
  // then shift the output range using the zero point value.
  //
  // N.B. Assumes MXCSR has been configured with the default rounding mode of
  // "round to nearest even".
  auto IntegerVector = _mm512_cvtps_epi32(FloatVector);
  IntegerVector = _mm512_add_epi32(IntegerVector, ZeroPointVector);

  return IntegerVector;
}

template <typename OutputType>
void
MlasQuantizeLinearPackBytes(
    __m512i IntegerVector,
    OutputType* base_addr);

template <>
MLAS_FORCEINLINE
void
MlasQuantizeLinearPackBytes<uint8_t>(
    __m512i IntegerVector,
    uint8_t* base_addr) {
  // to do: signed to uint8
  _mm512_mask_cvtusepi32_storeu_epi8(base_addr, 0xFFFF, IntegerVector);
}

template <>
MLAS_FORCEINLINE
void
MlasQuantizeLinearPackBytes<int8_t>(
    __m512i IntegerVector,
    int8_t* base_addr) {
  _mm512_mask_cvtsepi32_storeu_epi8(base_addr, 0xFFFF, IntegerVector);
}

template <typename OutputType>
void
MLASCALL
MlasQuantizeLinearAVX512(
    const float* Input,
    OutputType* Output,
    size_t N,
    float Scale,
    OutputType ZeroPoint)
/*++

Routine Description:

    This routine quantizes the input buffer using the supplied quantization
    parameters.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

    Scale - Supplies the quantization scale.

    ZeroPoint - Supplies the quantization zero point value.

Return Value:

    None.

--*/
{
  constexpr int32_t MinimumValue = std::numeric_limits<OutputType>::min();
  constexpr int32_t MaximumValue = std::numeric_limits<OutputType>::max();

  auto ScaleVector = _mm512_set1_ps(Scale);
  auto MinimumValueVector = _mm512_set1_ps(float(MinimumValue - ZeroPoint));
  auto MaximumValueVector = _mm512_set1_ps(float(MaximumValue - ZeroPoint));
  auto ZeroPointVector = _mm512_set1_epi32(ZeroPoint);

  while (N >= 16) {
    auto FloatVector = _mm512_loadu_ps(Input);
    auto IntegerVector = MlasQuantizeLinearVector(FloatVector, ScaleVector,
                                                  MinimumValueVector, MaximumValueVector, ZeroPointVector);

    MlasQuantizeLinearPackBytes<OutputType>(IntegerVector, Output);

    Input += 16;
    Output += 16;
    N -= 16;
  }

  for (size_t n = 0; n < N; n++) {
      float FloatValue = std::nearbyintf(Input[n] / Scale) + float(ZeroPoint);
      FloatValue = std::max(FloatValue, float(MinimumValue));
      FloatValue = std::min(FloatValue, float(MaximumValue));
      Output[n] = (OutputType)(int32_t)FloatValue;
  }
}

void
    MLASCALL
    MlasQuantizeLinearU8KernalAVX512(
        const float* Input,
        uint8_t* Output,
        size_t N,
        float Scale,
        uint8_t ZeroPoint) {
  MlasQuantizeLinearAVX512<uint8_t>(Input, Output, N, Scale, ZeroPoint);
}

void
    MLASCALL
    MlasQuantizeLinearS8KernalAVX512(
        const float* Input,
        int8_t* Output,
        size_t N,
        float Scale,
        int8_t ZeroPoint) {
  MlasQuantizeLinearAVX512<int8_t>(Input, Output, N, Scale, ZeroPoint);
}
