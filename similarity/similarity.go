package similarity

import "math"

func Cosine(a []float64, b []float64) float64 {
	if len(a) != len(b) {
		return -1.0
	}

	return ProductDot(a, b) / (VectorMagnitude(a) * VectorMagnitude(b))
}

func VectorMagnitude(vector []float64) float64 {
	squareSums := 0.0

	for _, v := range vector {
		squareSums += v * v
	}

	return math.Sqrt(squareSums)
}

func ProductDot(vectorA []float64, vectorB []float64) float64 {
	if len(vectorA) != len(vectorB) {
		return -1.0
	}

	result := 0.0
	for i := 0; i < len(vectorA); i++ {
		result += vectorA[i] * vectorB[i]
	}

	return result
}
