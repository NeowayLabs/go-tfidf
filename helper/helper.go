package helper

func RemoveDuplicates(words []string) []string {
	uniqueWords := make([]string, 0)

	keys := make(map[string]bool)
	for _, w := range words {
		if _, exists := keys[w]; !exists {
			uniqueWords = append(uniqueWords, w)
			keys[w] = true
		}
	}

	return uniqueWords
}

func StringArrayContainsWord(words []string, word string) bool {
	for _, w := range words {
		if word == w {
			return true
		}
	}

	return false
}
