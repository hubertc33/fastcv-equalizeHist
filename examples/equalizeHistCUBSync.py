import cv2
import torch
import fastcv


img = cv2.imread("../artifacts/rekin.jpg", cv2.IMREAD_GRAYSCALE)
img_tensor = torch.from_numpy(img).cuda()

equalized_tensor = fastcv.equalizeHistCUBSync(img_tensor)
equalized_image = equalized_tensor.cpu().numpy()
cv2.imwrite("../result/output_equalizeHistCUBSync.jpg", equalized_image)

print("saved equalized image.")
