# Push Final lên justeg4n/GRASP

Thực hiện từ PowerShell tại `D:\FPTU\Semester_8\REL301m\Final`.

## Trạng thái chuẩn bị

- Git chung được khởi tạo tại `Final`, nhánh `main`, remote `origin` trỏ tới `https://github.com/justeg4n/GRASP.git`.
- Metadata Git cũ của năm thành phần được chuyển vào `.git-backups/<tên-thành-phần>/.git`. Lịch sử và cấu hình cũ vẫn còn ở máy; thư mục backup được bỏ qua khi commit.
- Mã nguồn được quản lý như các thư mục thông thường, không phải submodule. Commit mới ở repository chung không chứa lịch sử commit của các repository cũ.
- Chưa tạo commit và chưa push. README của mỗi thành phần vẫn được giữ lại; các workflow trong `.github` của thành phần không tự chạy từ repository chung.

## Kiểm tra trước khi commit

```powershell
Set-Location 'D:\FPTU\Semester_8\REL301m\Final'
git remote -v
git status --short
git add .
git diff --cached --stat
git diff --cached --check
```

Kiểm tra danh sách staged để bảo đảm không có cấu hình riêng hoặc dữ liệu ngoài ý muốn. File mẫu `.env.example`, `config/vast.env.example`, splits, schema và bộ canonical gold evidence được giữ lại có chủ đích.

Mã nguồn cũ có thể có cảnh báo whitespace từ `git diff --cached --check`; kiểm tra vị trí được báo trước khi sửa. Không tự chuẩn hóa nội dung bộ canonical gold evidence vì manifest kiểm tra SHA256 theo byte.

Nếu Git yêu cầu danh tính, đặt tên/email dùng cho commit tại repository này:

```powershell
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_EMAIL"
```

## Nếu repository GitHub đang trống

Kiểm tra remote; lệnh có thể yêu cầu đăng nhập nếu repository là private:

```powershell
git ls-remote origin
```

Nếu lệnh thành công và không liệt kê ref nào, tạo commit rồi push:

```powershell
git commit -m "chore: prepare GRASP monorepo"
git push -u origin main
```

## Nếu GitHub đã có commit

Tạo commit local, tải lịch sử remote và xem tên nhánh hiện có:

```powershell
git commit -m "chore: prepare GRASP monorepo"
git fetch origin
git branch -r
```

Nếu nhánh remote là `origin/main`, hợp nhất lịch sử trước khi push:

```powershell
git merge origin/main --allow-unrelated-histories
```

Nếu có conflict (ví dụ README), sửa file để giữ nội dung cần thiết, rồi:

```powershell
git add .
git commit
git push -u origin main
```

Nếu merge hoàn tất tự động, chỉ cần lệnh push. Thay `origin/main` bằng nhánh thực tế nếu remote dùng tên khác. Không cần force push.

## Khôi phục Git riêng của một thành phần

Nếu sau này cần khôi phục repository con, ví dụ `grasp`, và `grasp/.git` chưa tồn tại:

```powershell
Move-Item -LiteralPath '.git-backups/grasp/.git' -Destination 'grasp/.git'
```

Thao tác này đưa thành phần trở lại dạng repository lồng nhau; chỉ thực hiện khi chủ động quay về cách quản lý cũ. Giữ `.git-backups` ở máy nếu còn cần lịch sử gốc.
