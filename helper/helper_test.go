package helper_test

import (
	"testing"

	"github.com/NeowayLabs/go-tfidf/helper"
	"github.com/stretchr/testify/assert"
)

func TestStringArrayContainsWordShouldReturnTrue(t *testing.T) {
	input := []string{"cake", "pizza", "avocado"}

	actual := helper.StringArrayContainsWord(input, "avocado")

	assert.True(t, actual)

}

func TestStringArrayContainsWordShouldReturnFalse(t *testing.T) {
	input := []string{"cake", "pizza", "avocado"}

	actual := helper.StringArrayContainsWord(input, "pie")

	assert.False(t, actual)

}

func TestRemoveDuplicatesShouldRemoveDuplicatedWords(t *testing.T) {
	input := []string{"cake", "pizza", "avocado", "pizza", "avocado", "pie"}
	expected := []string{"cake", "pizza", "avocado", "pie"}

	actual := helper.RemoveDuplicates(input)

	assert.Equal(t, expected, actual)

}

func TestRemoveDuplicatesShouldReturnSameArray(t *testing.T) {
	input := []string{"cake", "pizza", "avocado", "pie"}
	expected := []string{"cake", "pizza", "avocado", "pie"}

	actual := helper.RemoveDuplicates(input)

	assert.Equal(t, expected, actual)

}
