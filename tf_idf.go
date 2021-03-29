package tfidf

type TfIdf struct {
	Documents []string
	DocumentsTermFrequency []float64
	DocumentsInverseFrequency []float64

}

func New() *TfIdf {
	return &TfIdf{	
		Documents: make([]string, 0)
	}
}
