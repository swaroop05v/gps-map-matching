# Dark map and evaluator UI evidence

The original captures are in the parent directory. The screenshots here show the redesigned Folium maps and Streamlit UI using the same trips.

| Capture | What it demonstrates |
| --- | --- |
| [EDA trip 01](map_eda_train_00000001.png) | Full GPS trace, road network, and reference-route alignment. |
| [EDA trip 02](map_eda_train_00000002_offroute.png) | Raw GPS points outside the reference route. |
| [EDA trip 04](map_eda_train_00000004.png) | A second GPS, network, and reference-route overlay at trip scale. |
| [Largest gain, full route](map_result_largest_gain_00000077.png) and [zoom](map_result_zoom_largest_gain_00000077.png) | Where HMM changes the classifier-only route for trip 77. |
| [Median gain, full route](map_result_median_gain_00000007.png) and [zoom](map_result_zoom_median_gain_00000007.png) | Trip 07 has a stretch where different predicted arcs can overlap visually. |
| [Smallest gain, full route](map_result_smallest_gain_00000040.png) and [zoom](map_result_zoom_smallest_gain_00000040.png) | Where HMM changes the classifier-only route for trip 40, the smallest-gain example. |
| [Live metrics](ui_1_metrics.jpg) | The trip selector and server-computed metrics after a real API run for student-test trip 77. |
| [Live map comparison](ui_2_map.jpg) | The same API result with the classifier-only layer enabled. |

Interactive HTML versions of the offline screenshots are in [maps/](maps/). The report maps include the reference road layer. In the evaluator UI, reference routes stay on the API laptop.

To regenerate the offline evidence from the repository root, run `python scripts/06_result_maps.py --gta`. Add `--maps-only` to write the themed interactive maps without browser capture.

## Submission checks

- The offline Folium maps include GPS observations, road geometry, reference routes and, for result maps, predicted routes. The maps have legends, layer controls, and both full-trip and zoomed examples.
- The existing technical report includes representative geographic maps, original UI captures, and API log excerpts. Use the redesigned screenshots in the deployment section when preparing the final report.
- The rubric separately requires a screenshot of a successful API response in the receiving laptop's terminal. The checked-in log excerpts show request results. Capture the terminal during the two-laptop demonstration for the required screenshot.
- Student-test trips were available for validation. The reserved evaluator trips and a two-laptop deployment still need to be checked at the final evaluation.
