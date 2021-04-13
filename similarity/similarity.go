// Package similarity provides similarity algorithm implementations
package similarity

import (
	"errors"
	"math"
	"strings"
)

// CalculateSimilarities receives the query individual tf-idf, query tf-idf computed for every documents in *TfIdf object, and a string with the desired similarity function to be used.
// The default similarity function is Cosine.
// If the TfIdf vectors have different lengths, returns an error.
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
			err = errors.New("vectors have different lengths")
			break
		}
		similarities = append(similarities, similarity)
	}

	return similarities, err
}

// Cosine implements the Cosine similarity algorithm.
// Receives two vectors for computing its similarity.
// If the vector lengths are different, the function returns -1.0, indicating error.
func Cosine(a []float64, b []float64) float64 {
	if len(a) != len(b) {
		return -1.0
	}

	magnitudes := vectorMagnitude(a) * vectorMagnitude(b)
	if magnitudes > 0.0 {
		return productDot(a, b) / magnitudes
	}

	return 0.0
}

func vectorMagnitude(vector []float64) float64 {
	squareSums := 0.0

	for _, v := range vector {
		squareSums += v * v
	}

	return math.Sqrt(squareSums)
}

func productDot(vectorA []float64, vectorB []float64) float64 {
	if len(vectorA) != len(vectorB) {
		return -1.0
	}

	result := 0.0
	for i := 0; i < len(vectorA); i++ {
		result += vectorA[i] * vectorB[i]
	}

	return result
}
