package helper

func RemoveDuplicates(words []string) []string {
	uniqueWords := make([]string, 0)

	for _, w := range words {
		if !StringArrayContainsWord(uniqueWords, w) {
			uniqueWords = append(uniqueWords, w)
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
