from openlayer.lib.core import metrics
from openlayer.types.inference_pipelines import data_stream_params

class Metric(metrics.BaseMetric):
    """Computes a simple capitalization/format check.
       # Akhil Custom Metric:
       Checks if the chatbot response:
       - Starts with an uppercase letter (optionally after quotes)
       - Ends with acceptable punctuation like ., ?, !, or their quoted forms
       Returns 1.0 if both are true, else 0.0.
    """

    def compute_on_dataset(self, dataset: metrics.Dataset) -> metrics.MetricReturn:
        """Compute the metric over the full dataset."""
        # Akhil Custom Metric: Apply the per-row check to the dataset
        dataset.df["score"] = dataset.df.apply(
            lambda x: self.compute_on_row(x, dataset.config), axis=1
        )
        score = dataset.df["score"].mean() if len(dataset.df) > 0 else 0.0

        # Akhil Custom Metric: Return the mean score across dataset
        return metrics.MetricReturn(
            value=score,
            unit=None,
            meta=None,
            added_cols={"score"},
        )

    def compute_on_row(
        self, data: dict, config: data_stream_params.ConfigLlmData
    ) -> float:
        """Per-row capitalization/format check."""
        # Akhil Custom Metric: Safely fetch model output and normalize
        output_col = config["outputColumnName"]
        output = data.get(output_col, "")
        if output is None:
            output = ""
        output = str(output).strip()

        # Akhil Custom Metric: Handle empty responses
        if not output:
            return 0.0

        # Akhil Custom Metric: Allow leading quote before first letter
        first_char = output[0]
        if first_char in {'"', "'"} and len(output) > 1:
            first_char = output[1]

        # Akhil Custom Metric: Check capitalization and punctuation
        starts_capitalized = first_char.isalpha() and first_char.isupper()
        valid_endings = {".", "?", "!", '."', '?"', '!"', ".'", "?’", "!’"}
        ends_with_punct = any(output.endswith(e) for e in valid_endings)

        # Akhil Custom Metric: Score 1 if both conditions satisfied
        return 1.0 if (starts_capitalized and ends_with_punct) else 0.0


# Don't change this
if __name__ == "__main__":
    Metric().run()
