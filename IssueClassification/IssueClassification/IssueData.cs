using Microsoft.ML.Data;

namespace IssueClassification
{
    public class Issue
    {
        [LoadColumn(0)]
        public string ID { get; set; }
        [LoadColumn(1)]
        public string Area { get; set; }
        [LoadColumn(2)]
        public string Title { get; set; }
        [LoadColumn(3)]
        public string Description { get; set; }
    }

    public class IssuePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Area;
    }
}
