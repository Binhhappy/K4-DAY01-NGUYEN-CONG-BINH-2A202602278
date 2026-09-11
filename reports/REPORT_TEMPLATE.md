# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 2026-09-11

**Runtime Colab:** Không dùng Colab – chạy local trong `.venv` (Windows 11), device `cpu` (`torch.cuda.is_available() == False`)

**Python / PyTorch / Ultralytics:** Python 3.10.11 / PyTorch 2.14.0+cpu / Ultralytics 8.4.145 (pin `ultralytics==8.4.145` trong notebook và `requirements.txt`)

**Checkpoint:** `yolo11n-cls.pt` (sha256 `c62d41bf…2bd7`), `yolo11n.pt` (sha256 `0ebbc80d…4ee1`), `yolo11n-seg.pt` (sha256 `55ed65c5…b152`) – tải từ `ultralytics/assets v8.4.0`, checksum khớp

**Sample / checksum:** `traffic` (COCO 210273, 640×428), `kitchen` (COCO 397133, 640×427), `dining` (COCO 166918, 480×640) – tải từ `images.cocodataset.org/val2017`, sha256 khớp ô setup. Threshold detection = segmentation = 0.35.

**Thay đổi so với notebook nguồn:** Không thay đổi mã. Chỉ chạy lại ô so sánh threshold (0.20 / 0.35 / 0.60) cho sample `kitchen` bằng đúng `.venv` trên để lấy số liệu cho mục 2; ô Google Drive in `SKIP` vì không chạy trên Colab.

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):

  ```json
  {"sample_id": "traffic", "coco_image_id": 210273, "task": "image_classification",
   "taxonomy_name": "ImageNet-1K", "model_file": "yolo11n-cls.pt",
   "rank": 1, "class_id": 468, "class_name": "cab", "score": 0.510915}
  ```

  Top-5 đầy đủ của `traffic`: `cab` 0.511 → `minibus` 0.164 → `police_van` 0.086 → `recreational_vehicle` 0.054 → `streetcar` 0.048 (xem `visuals/classification_top5.png`).
- Record này mô tả toàn ảnh như thế nào? Record gán **một nhãn cho toàn bộ ảnh**: mô hình cho rằng ảnh đường phố này "là ảnh về taxi (cab)" với score 0.51. Không có tọa độ, không cho biết taxi ở đâu hay có bao nhiêu chiếc; các nhãn hạng 2–5 đều là phương tiện giao thông, cho thấy mô hình chỉ nắm được "chủ đề" chung của ảnh. Thực tế ảnh có nhiều xe buýt, ô tô và người (theo detection ở mục 2), nhưng phân loại chỉ tóm gọn thành một lớp.
- Ai định nghĩa class list mà checkpoint có thể dự đoán? Class list là taxonomy **ImageNet-1K** (1000 lớp) do bộ dữ liệu ImageNet định nghĩa từ trước; checkpoint `yolo11n-cls.pt` được Ultralytics huấn luyện trên taxonomy đó nên chỉ dự đoán được đúng 1000 lớp này. Mô hình không tự "phát minh" lớp mới; nếu dự án cần lớp khác (ví dụ "giao lộ đông xe"), guideline/taxonomy của dự án phải định nghĩa và gán nhãn lại.
- Vì sao cần giữ cả ID, tên lớp và tên taxonomy? Vì `class_id = 468` chỉ có nghĩa trong taxonomy ImageNet-1K; cùng số 468 ở COCO-80 hoặc taxonomy dự án sẽ là lớp khác (hoặc không tồn tại). Tên lớp giúp người đọc kiểm tra bằng mắt và tránh nhầm khi ID bị lệch (off-by-one, đổi thứ tự), còn ID giúp máy so khớp chính xác không phụ thuộc chính tả/ngôn ngữ. Giữ cả ba trường cùng `model_file`/`model_sha256` đảm bảo nhãn có thể truy vết và tái lập.
- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì? Guideline phải quy định **tiêu chí chọn nhãn chính** khi ảnh có nhiều chủ thể: ví dụ chọn theo vật thể chiếm diện tích lớn nhất, theo mục đích dự án (cảnh "đường phố" thay vì từng xe), hoặc quy định ưu tiên khi có xung đột (`bus` vs `cab`); đồng thời quy định có cho phép đa nhãn (multi-label) hay không, và trường hợp nào phải đánh dấu "không xác định/escalate". Với ảnh `traffic`, không có quy tắc thì annotator A có thể ghi `bus`, B ghi `cab`, C ghi `street scene` – nhãn không nhất quán.
- Vì sao model score không phải ground truth? Score 0.51 chỉ là độ tin cậy nội bộ của mô hình (softmax) dựa trên dữ liệu nó đã học, không phải xác nhận của con người theo guideline dự án. Mô hình có thể tự tin mà vẫn sai (ảnh `kitchen` được dự đoán là `gong` 0.42 – rõ ràng sai) và có thể đúng với score thấp. Ground truth phải do người gán theo guideline và được reviewer xác nhận; score chỉ dùng để sắp xếp ưu tiên xem xét, không dùng làm nhãn.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):

  ```json
  {"sample_id": "kitchen", "coco_image_id": 397133, "image_width": 640, "image_height": 427,
   "task": "object_detection", "taxonomy_name": "COCO-80", "model_file": "yolo11n.pt",
   "score_threshold": 0.35, "class_id": 0, "class_name": "person", "score": 0.912625,
   "coordinate_unit": "pixel", "bbox_format": "xyxy",
   "bbox_xyxy": [385.33, 69.24, 498.92, 348.92], "bbox_width": 113.58, "bbox_height": 279.68}
  ```

  Ở threshold 0.35, `kitchen` có 11 prediction: person ×2, bowl ×5, oven ×2, cup ×2 (toàn bộ 3 sample: 53 record).
- Diễn giải vị trí box bằng lời: Gốc tọa độ ở góc trên-trái, đơn vị pixel. Box bắt đầu tại x = 385 (khoảng 60% chiều rộng ảnh 640) và kết thúc tại x = 499 (≈78%); theo chiều dọc từ y = 69 (≈16% chiều cao 427) xuống y = 349 (≈82%). Tức là box nằm ở **nửa bên phải ảnh, kéo dài gần hết chiều cao**, rộng 114 px và cao 280 px (cao gấp ~2,5 lần rộng) – đúng với người đầu bếp đứng quay lưng trước bếp trong ảnh minh họa. Box thứ hai `person` 0.61 tại `[0.12, 263.18, 61.55, 310.78]` chỉ là cánh tay ở mép trái ảnh, phần thân bị cắt ngoài khung.
- So sánh số prediction ở hai threshold: Chạy lại `run_detection` cho `kitchen`:

  | threshold          | số vật thể | các lớp                                                       |
  | ------------------ | ------------- | --------------------------------------------------------------- |
  | 0.20               | 17            | 11 lớp ở 0.35 + spoon ×3, potted plant, dining table, bottle |
  | 0.35 (mặc định) | 11            | person ×2, bowl ×5, oven ×2, cup ×2                         |
  | 0.60               | 6             | person ×2, bowl ×2, oven ×2                                  |

  Hạ ngưỡng từ 0.35 xuống 0.20 tăng từ 11 lên 17 prediction (+6); nâng lên 0.60 giảm còn 6 (−5). Các lớp mất đi ở ngưỡng cao là bát/cốc nhỏ, thìa, chậu cây – những vật thể nhỏ hoặc bị che.
- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem? Ngưỡng thấp cho **độ bao phủ cao hơn** (bắt thêm thìa, chậu cây, bàn ăn – đều là vật thật trong ảnh) nhưng **reviewer phải xem nhiều box hơn** và tỉ lệ box sai/trùng tăng (ở 0.20 xuất hiện `bottle` chưa chắc đúng, và box `oven` 0.69 bao trùm cả bàn bếp phía trước). Ngưỡng cao giảm khối lượng review nhưng bỏ sót nhiều vật thể (5 bát/cốc mất đi ở 0.60). Vì vậy threshold là tham số **pre-label/ưu tiên review**, không phải quy tắc ground truth: annotator vẫn phải vẽ box cho mọi vật thể trong phạm vi guideline dù mô hình không phát hiện (ví dụ hàng chục nồi chảo treo trên tường hoàn toàn không có prediction ở mọi ngưỡng).
- Đề xuất một quy tắc box chặt: "Box phải ôm sát phần **nhìn thấy** của vật thể: cạnh box chạm điểm ngoài cùng của vật thể ở 4 phía, sai số ≤ 2 px (hoặc ≤ 2% cạnh box, lấy giá trị lớn hơn); không bao gồm bóng, tay cầm của vật khác hay nền; một box chỉ chứa một instance." Áp dụng vào quan sát: box `oven` 0.69 `[0.11, 187.54, 195.96, 292.81]` đang bao cả mặt bếp lẫn phần bàn và các bát phía trước – theo quy tắc này phải thu lại chỉ còn thân lò, các bát tách thành box riêng.
- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định? **Guideline phải quy định trước**: (1) vẽ box theo phần nhìn thấy hay theo hình dạng ước lượng đầy đủ (amodal); (2) ngưỡng che khuất tối thiểu để vẫn gán nhãn (ví dụ còn thấy ≥ 20% và nhận ra được lớp); (3) vật thể cắt mép ảnh thì box chạm sát mép, có cờ `truncated`; (4) vật thể quá nhỏ (ví dụ < 10×10 px như `bowl` 0.46 kích thước 18×17 px trên kệ) có gán hay bỏ. **Cần escalation** khi annotator không thể xác định lớp hoặc số instance từ phần nhìn thấy: cánh tay `person` 0.61 ở mép trái (chỉ thấy tay – có tính là người không?), cụm bát chồng nhau `bowl` 0.38/0.50/0.72 ở góc trái (bao nhiêu bát, cái nào là bát, cái nào là khay?) – những ca này ghi vào hàng đợi hỏi lead/QC thay vì tự đoán.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):

  ```json
  {"sample_id": "kitchen", "task": "instance_segmentation", "taxonomy_name": "COCO-80",
   "model_file": "yolo11n-seg.pt", "score_threshold": 0.35,
   "instance_id": "kitchen-001", "class_id": 0, "class_name": "person", "score": 0.899318,
   "coordinate_unit": "pixel", "bbox_format": "xyxy", "bbox_xyxy": [385.45, 66.44, 498.02, 348.58],
   "polygon_point_count": 348,
   "polygon_xy": [[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0], "..."]}
  ```

  `kitchen` có 11 instance (`kitchen-001` … `kitchen-011`): person ×2, bowl ×4, potted plant, dining table (598 điểm), oven (213 điểm), spoon ×2 (46 và 37 điểm); bát nhỏ nhất `kitchen-010` chỉ có 12 điểm. Toàn bộ 3 sample: 49 instance.
- Polygon bổ sung chi tiết gì so với box? Box của `kitchen-001` chỉ là hình chữ nhật 113×282 px, trong đó có nhiều pixel nền (tường, kệ, nồi treo phía sau). Polygon 348 điểm đi theo **đường viền thực** của người: mái tóc, hai cánh tay, dây đeo tạp dề, chân tách ra ở dưới – cho biết chính xác pixel nào thuộc người, pixel nào là nền. Với vật thể không phải hình chữ nhật (`potted plant` lá rủ, `dining table` mặt bàn hình thang bị người/bát che một phần), polygon loại bỏ được phần nền mà box buộc phải bao gồm. Ngược lại, polygon tốn nhiều điểm hơn hẳn (598 điểm cho bàn) nên gán nhãn và QC khó, chậm hơn.
- `instance_id` dùng để làm gì và không phải loại ID nào? `instance_id` (`kitchen-001`, `kitchen-002`, …) **phân biệt từng đối tượng riêng lẻ trong một ảnh** để mỗi polygon/mask gắn với đúng một đối tượng, kể cả khi cùng lớp: 4 bát có 4 `instance_id` khác nhau nhưng cùng `class_id = 45`. Nó **không phải** `class_id` (mã lớp trong taxonomy – dùng chung cho mọi bát) và **không phải** track ID (ID theo dõi cùng một đối tượng qua nhiều khung hình video – ở đây mỗi ảnh độc lập, `kitchen-001` không liên quan `traffic-001`). Ô validation kiểm tra `instance_id` phải duy nhất trong toàn bộ output.
- Đề xuất một quy tắc biên mask: "Biên mask bám theo cạnh nhìn thấy của đối tượng với sai lệch ≤ 2 px; đặt đỉnh polygon tại mọi điểm đổi hướng rõ rệt, khoảng cách giữa hai đỉnh liên tiếp trên đoạn cong ≤ 5 px; **không** bao gồm bóng đổ, vùng nền lọt giữa (ví dụ khoảng trống giữa hai chân, giữa cánh tay và thân); hai instance chạm nhau phải có biên chung không chồng lấn, không để hở." Đối chiếu bằng mắt: mask `dining table` 0.58 hiện tràn xuống cả thùng/chân bàn phía dưới và các bát trên bàn; mask `oven` 0.49 phủ cả mảng khay trên bàn – đều vi phạm quy tắc và cần rework.
- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định? **Guideline quy định**: (1) vùng mờ chuyển động/mất nét thì lấy biên tại điểm chuyển đổi 50% giữa vật thể và nền; (2) hai instance tiếp xúc (bát chồng bát, tạp dề trên người) thì mask theo phần nhìn thấy, phần bị che thuộc về instance phía trước; (3) lỗ thủng nhìn xuyên nền có tính vào mask hay không; (4) instance bị cắt mép thì polygon chạy sát mép ảnh. **Escalation** khi annotator không xác định được ranh giới hay danh tính: mảng `oven` 0.49 trùng với `dining table` 0.58 ở góc trái (mặt bếp kết thúc ở đâu, mặt bàn bắt đầu ở đâu?), `bowl` 0.53 với `bowl` 0.70 chồng lên nhau (một bát hay hai bát?), `person` 0.43 chỉ có cánh tay – các ca này gắn cờ `needs_review` và chuyển lead/QC quyết định, không tự đoán biên.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ              | Đơn vị/định dạng ground truth                                                                                                                              | Lỗi hoặc điểm mơ hồ quan sát được                                                                                                                                                                                                                                                              | Annotator làm gì?                                                                                                                                                                                                       | Reviewer xem gì?                                                                                                                                                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Phân loại ảnh      | 1 nhãn/ảnh:`class_id` + `class_name` thuộc taxonomy đã khai báo (`taxonomy_name`), kèm `sample_id`; không có tọa độ, không lưu score       | `kitchen` bị dự đoán `gong` 0.42 (sai hoàn toàn); `traffic` là `cab` 0.51 dù ảnh nhiều xe buýt hơn; `dining` là `restaurant` 0.79 nhưng cảnh có thể là "quán rượu/quầy bar" – taxonomy ImageNet không có lớp phù hợp tuyệt đối                                | Xem toàn ảnh, chọn đúng một lớp theo tiêu chí ưu tiên trong guideline; nếu ảnh nhiều chủ thể hoặc không có lớp phù hợp thì gắn cờ hỏi thay vì đoán; không nhìn score mô hình để chọn  | Nhãn có nằm trong taxonomy không; có tuân tiêu chí chọn nhãn chính không; kiểm tra ngẫu nhiên và các ca score thấp/lệch giữa top-1 và nhãn người; đo độ đồng thuận giữa annotator                      |
| Phát hiện vật thể | N box/ảnh, mỗi box:`class_id`, `bbox_xyxy` pixel (gốc trên-trái), cờ `truncated`/`occluded`; không lưu score                                     | Box`oven` 0.69 bao cả bàn + bát; `person` 0.61 chỉ là cánh tay cắt mép; nhiều nồi chảo treo tường không được phát hiện ở mọi ngưỡng; hạ threshold 0.35→0.20 tăng 11→17 box, thêm `bottle` chưa chắc đúng                                                           | Vẽ box cho**mọi** vật thể trong phạm vi guideline kể cả mô hình bỏ sót; box ôm sát phần nhìn thấy (sai số ≤ 2 px); sửa box pre-label quá rộng; gắn cờ ca che khuất/quá nhỏ theo guideline | Thiếu box (so với ảnh, không so với prediction); box lỏng/lệch lớp; box trùng cho một vật thể; cờ truncated/occluded đúng chưa; ưu tiên xem các box bị annotator sửa nhiều và ảnh có mật độ vật thể cao |
| Instance segmentation | N polygon/ảnh, mỗi instance:`instance_id` duy nhất, `class_id`, `polygon_xy` pixel khép kín (≥ 3 điểm), bbox suy ra từ polygon; không lưu score | Mask`dining table` 0.58 tràn xuống thùng và chân bàn, chồng lên các bát; `oven` 0.49 và `dining table` chồng lấn vùng góc trái; `bowl` 0.53/0.70 chồng nhau; `kitchen-010` chỉ 12 điểm cho bát nhỏ – biên thô; mask `person` có dây tạp dề bị nuốt vào thân | Vẽ polygon theo biên thực (≤ 2 px), mỗi instance một`instance_id`, tách bát/thìa/bàn thành instance riêng, không chồng lấn; các ca ranh giới mờ/tiếp xúc gắn cờ escalation                        | Biên có bám vật thể không (zoom vào cạnh), có chồng lấn/hở giữa instance không,`instance_id` có trùng không, số instance có khớp số vật thể nhìn thấy; kiểm tra bằng overlay mask lên ảnh gốc         |

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu: Chỉ dùng ảnh có nguồn và giấy phép rõ ràng (ở bài này: 3 ảnh COCO val2017, CC BY 2.0, ghi trong `IMAGE_ATTRIBUTION.md`, cố định bằng image ID + SHA-256). Không đưa ảnh cá nhân, ảnh khách hàng/VinFast, ảnh có mặt người hay biển số nhận dạng được vào repository công khai; không ghi họ tên, MSSV, email, số điện thoại trong báo cáo, output JSON hoặc tên tệp; output chỉ chứa metadata kỹ thuật (model, checksum, tọa độ).
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho: giảng viên/lead phụ trách bài lab (qua kênh VLearn hoặc kênh lớp chính thức), không tự xoá/sửa/chia sẻ tiếp dữ liệu đó, không commit lên repository, và ghi lại `sample_id`/đường dẫn tệp để người phụ trách xử lý.

## 6. Danh sách bằng chứng

- [X] `classification_predictions.json` (15 record: 3 sample × top-5)
- [X] `detection_predictions.json` (53 record, threshold 0.35)
- [X] `segmentation_predictions.json` (49 instance, threshold 0.35)
- [X] `IMAGE_ATTRIBUTION.md`
- [X] `visuals/classification_top5.png`
- [X] `visuals/detection_predictions.png`
- [X] `visuals/segmentation_prediction.png`
- [X] Ô validation cuối notebook báo `PASS` (đã tạo `day1_lab_outputs.zip`, 5,6 MB).
- [X] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
