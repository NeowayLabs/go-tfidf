package similarity

import (
	"errors"
	"math"
	"strings"
)

func CalculateSimilarities(queryTfIdf []float64, queryTfIdfDocuments [][]float64, similarityFunction string) ([]float64, error) {
	var err error
	similarities := make([]float64, 0)

	var similarityFunctionToBeCalled interface{}

	switch strings.ToLower(similarityFunction) {
	default:
		similarityFunctionToBeCalled = Cosine
	}

	for _, docQueryTfidf := range queryTfIdfDocuments {
		similarity := similarityFunctionToBeCalled.(func([]float64, []float64) float64)(queryTfIdf, docQueryTfidf)
		if similarity < 0.0 {
			similarities = make([]float64, 0)
			err = errors.New("Vectors have different lengths")
			break
		}
		similarities = append(similarities, similarity)
	}

	return similarities, err
}

func Cosine(a []float64, b []float64) float64 {
	if len(a) != len(b) {
		return -1.0
	}

	magnitudes := VectorMagnitude(a) * VectorMagnitude(b)
	if magnitudes > 0.0 {
		return ProductDot(a, b) / magnitudes
	}

	return 0.0
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
