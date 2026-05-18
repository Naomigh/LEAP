/*
 * vector_ed.c
 *
 *  Created on: Nov 8, 2013
 *      Author: hxin
 */

#include "shift.h"
#include <stdio.h>
#include <string.h>

__m128i shift_right_sse(__m128i vec, int shift_num) {
	if (shift_num >= 64) {
		vec = _mm_slli_si128(vec, 8);
		shift_num = shift_num % 64;
	}
	__m128i carryover = _mm_slli_si128(vec, 8);
	carryover = _mm_srli_epi64(carryover, 64 - shift_num);
	vec = _mm_slli_epi64(vec, shift_num);
	return _mm_or_si128(vec, carryover);
}

__m128i shift_left_sse(__m128i vec, int shift_num) {
	if (shift_num >= 64) {
		vec = _mm_srli_si128(vec, 8);
		shift_num = shift_num % 64;
	}
	__m128i carryover = _mm_srli_si128(vec, 8);
	carryover = _mm_slli_epi64(carryover, 64 - shift_num);
	vec = _mm_srli_epi64(vec, shift_num);
	return _mm_or_si128(vec, carryover);
}

__m256i shift_right_avx(__m256i vec, int shift_num) {
	if (shift_num >= 128) {
		vec = _mm256_inserti128_si256(_mm256_setzero_si256(), _mm256_extracti128_si256(vec, 0), 1);
		shift_num = shift_num % 128;
	}
	if (shift_num >= 64) {
		vec = _mm256_slli_si256(vec, 8);
		shift_num = shift_num % 64;
	}
	__m256i carryover = _mm256_slli_si256(vec, 8);
	carryover = _mm256_srli_epi64(carryover, 64 - shift_num);
	vec = _mm256_slli_epi64(vec, shift_num);
	return _mm256_or_si256(vec, carryover);
}

__m256i shift_left_avx(__m256i vec, int shift_num) {
	if (shift_num >= 128) {
		vec = _mm256_inserti128_si256(_mm256_setzero_si256(), _mm256_extracti128_si256(vec, 1), 0);
		shift_num = shift_num % 128;
	}
	if (shift_num >= 64) {
		vec = _mm256_srli_si256(vec, 8);
		shift_num = shift_num % 64;
	}
	__m256i carryover = _mm256_srli_si256(vec, 8);
	carryover = _mm256_slli_epi64(carryover, 64 - shift_num);
	vec = _mm256_srli_epi64(vec, shift_num);
	return _mm256_or_si256(vec, carryover);
}

#ifdef USE_AVX512
static __m512i shift_qwords_up_avx512(__m512i vec, int words) {
	if (words <= 0)
		return vec;
	if (words >= 8)
		return _mm512_setzero_si512();

	__m512i idx = _mm512_set_epi64(7 - words, 6 - words, 5 - words, 4 - words,
			3 - words, 2 - words, 1 - words, 0);
	__mmask8 mask = (uint8_t)(0xffu << words);
	return _mm512_maskz_permutexvar_epi64(mask, idx, vec);
}

static __m512i shift_qwords_down_avx512(__m512i vec, int words) {
	if (words <= 0)
		return vec;
	if (words >= 8)
		return _mm512_setzero_si512();

	__m512i idx = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
	idx = _mm512_add_epi64(idx, _mm512_set1_epi64(words));
	__mmask8 mask = (uint8_t)(0xffu >> words);
	return _mm512_maskz_permutexvar_epi64(mask, idx, vec);
}

__m512i shift_right_avx512(__m512i vec, int shift_num) {
	if (shift_num <= 0)
		return vec;

	vec = shift_qwords_up_avx512(vec, shift_num / 64);
	shift_num %= 64;
	if (shift_num == 0)
		return vec;

	__m512i carryover = shift_qwords_up_avx512(vec, 1);
	carryover = _mm512_srli_epi64(carryover, 64 - shift_num);
	vec = _mm512_slli_epi64(vec, shift_num);
	return _mm512_or_si512(vec, carryover);
}

__m512i shift_left_avx512(__m512i vec, int shift_num) {
	if (shift_num <= 0)
		return vec;

	vec = shift_qwords_down_avx512(vec, shift_num / 64);
	shift_num %= 64;
	if (shift_num == 0)
		return vec;

	__m512i carryover = shift_qwords_down_avx512(vec, 1);
	carryover = _mm512_slli_epi64(carryover, 64 - shift_num);
	vec = _mm512_srli_epi64(vec, shift_num);
	return _mm512_or_si512(vec, carryover);
}
#endif
