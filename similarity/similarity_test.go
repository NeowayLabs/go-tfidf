package similarity_test

import (
	"testing"

	"github.com/NeowayLabs/go-tfidf/similarity"
	"github.com/stretchr/testify/assert"
)

func TestCalculateSimilaritiesWhenVectorsHaveDifferentLength(t *testing.T) {
	inputA := []float64{1.0, 2.0, 3.0, 1.0, 1.0}
	inputB := [][]float64{
		[]float64{1.0, 2.0, 3.0, 1.0},
	}
	expected := []float64{}

	actual, err := similarity.CalculateSimilarities(inputA, inputB, "Cosine")

	assert.Equal(t, expected, actual)
	assert.NotNil(t, err)
}

func TestCalculateSimilarities(t *testing.T) {
	inputA := []float64{1.0, 2.0, 3.0, 1.0, 1.0}
	inputB := [][]float64{
		[]float64{1.0, 2.0, 3.0, 1.0, 1.0},
	}
	expected := []float64{1.0}

	actual, err := similarity.CalculateSimilarities(inputA, inputB, "Cosine")

	assert.Equal(t, expected, actual)
	assert.Nil(t, err)
}

func TestVectorMagnitude(t *testing.T) {
	input := []float64{1.0, 2.0, 3.0, 1.0, 1.0}
	expected := 4.0
	actual := similarity.VectorMagnitude(input)

	assert.Equal(t, expected, actual)
}

func TestProductDotWhenVectorsHaveDifferentLength(t *testing.T) {
	inputA := []float64{1.0, 2.0, 3.0, 1.0, 1.0}
	inputB := []float64{1.0, 2.0, 3.0, 1.0}
	expected := -1.0
	actual := similarity.ProductDot(inputA, inputB)

	assert.Equal(t, expected, actual)
}

func TestProductDot(t *testing.T) {
	inputA := []float64{1.0, 2.0, 3.0, 1.0, 1.0}
	inputB := []float64{1.0, 2.0, 3.0, 1.0, 1.0}
	expected := 16.0
	actual := similarity.ProductDot(inputA, inputB)

	assert.Equal(t, expected, actual)
}

func TestCosineWhenVectorsHaveDifferentLength(t *testing.T) {
	inputA := []float64{1.0, 2.0, 3.0, 1.0, 1.0}
	inputB := []float64{1.0, 2.0, 3.0, 1.0}
	expected := -1.0
	actual := similarity.Cosine(inputA, inputB)

	assert.Equal(t, expected, actual)
}

func TestCosine(t *testing.T) {
	inputA := []float64{1.0, 2.0, 3.0, 1.0, 1.0}
	inputB := []float64{1.0, 2.0, 3.0, 1.0, 1.0}
	expected := 1.0
	actual := similarity.Cosine(inputA, inputB)

	assert.Equal(t, expected, actual)
}
