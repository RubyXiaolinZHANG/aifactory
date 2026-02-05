import os
import numpy as np
import cv2
from .image import RawInfo
from .scaling import normalization
from aifactory.utils.superscript_converter import SuperscriptConverter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def add_pq_roi_method(cls):

    def save_result(self, save_dir):
        # plot roi
        if isinstance(self.src_image, RawInfo):
            bgr = self.src_image.demosaic()
        else:
            bgr = self.src_image.image
        cv2.rectangle(bgr, (self.x1, self.y1), (self.x2, self.y2), (0, 255, 0), 2)
        label = self.type
        font_scale = 2
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        text_y = self.y1 - 10 if self.y1 - 10 > 10 else self.y2 + 20
        cv2.rectangle(bgr,
                      (self.x, text_y - text_height - 5),
                      (self.x + text_width, text_y + 5),
                      (0, 255, 0), -1)
        cv2.putText(bgr, label, (self.x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        os.makedirs(save_dir, exist_ok=True)
        save_name = os.path.join(save_dir, "0_roi_{}.png".format(self.type.replace(" ", "_")))
        cv2.imwrite(save_name, bgr)
        getattr(self, "save_{}".format(self.type.replace(" ", "_")))(save_dir)

    def save_solid_color(self, save_dir, with_index=1):
        os.makedirs(save_dir, exist_ok=True)
        # 1. save 3sigma bayer
        bayer = np.vstack((np.hstack((self.roi_image.bayer[:, :, 0], self.roi_image.bayer[:, :, 1])),
                           np.hstack((self.roi_image.bayer[:, :, 2], self.roi_image.bayer[:, :, 3]))))
        clip_min = self.roi_image._histogram['mean'] - 3 * self.roi_image._histogram['std']
        clip_max = self.roi_image._histogram['mean'] + 3 * self.roi_image._histogram['std']
        bayer_vis = np.round(normalization(bayer, clip_min, clip_max, 0, 255)).astype(np.uint8)
        font_scale = 0.5
        superscript_cvt = SuperscriptConverter()
        label = "N({:.3f}, {}), [{:.3f}, {:.3f}]".format(self.roi_image._histogram['mean'],
                                                             # self.roi_image._histogram['std'],
                                                       superscript_cvt.format_expression(
                                                           "{:.3f}".format(self.roi_image._histogram['std']), "2"),
                                                             clip_min, clip_max)
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        h_shift = text_height + 20
        save_h = bayer_vis.shape[0] + h_shift  # text_height + 20
        save_w = bayer_vis.shape[1] if bayer_vis.shape[1] > text_width else text_width
        save_image = np.zeros((save_h, save_w), dtype=np.uint8) + 255
        save_image[h_shift:, :bayer_vis.shape[1]] = bayer_vis
        # save_image = cv2.cvtColor(save_image, cv2.COLOR_GRAY2BGR)
        label = "N({:.3f}, {:.3f}".format(self.roi_image._histogram['mean'], self.roi_image._histogram['std'])
        (text_width, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.putText(save_image, label, (0, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        (text_width_superscript, _), _ = cv2.getTextSize("2", cv2.FONT_HERSHEY_SIMPLEX, font_scale/2, 2)
        cv2.putText(save_image, "2", (text_width, 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale/2, (0, 0, 0), 1, cv2.LINE_AA)
        label = "), [{:.3f}, {:.3f}]".format(clip_min, clip_max)
        cv2.putText(save_image, label, (text_width + text_width_superscript, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        std_val = self.roi_image.std(by_channel=True)
        label = "{:.3f}".format(std_val[0])
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(save_image,
                      (0, h_shift),
                      (text_width, h_shift + text_height + 10),
                      (255, 255, 255), -1)
        cv2.putText(save_image, label, (0, h_shift + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(save_image,
                      (self.roi_image.bayer_w, h_shift),
                      (self.roi_image.bayer_w + text_width, h_shift + text_height + 10),
                      (255, 255, 255), -1)
        cv2.putText(save_image, "{:.3f}".format(std_val[1]),
                    (self.roi_image.bayer_w, h_shift + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(save_image,
                      (0, self.roi_image.bayer_h + h_shift),
                      (text_width, self.roi_image.bayer_h + h_shift + text_height + 10),
                      (255, 255, 255), -1)
        cv2.putText(save_image, "{:.3f}".format(std_val[2]),
                    (0, self.roi_image.bayer_h + h_shift + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(save_image,
                      (self.roi_image.bayer_w, self.roi_image.bayer_h + h_shift),
                      (self.roi_image.bayer_w + text_width, self.roi_image.bayer_h + h_shift + text_height + 10),
                      (255, 255, 255), -1)
        cv2.putText(save_image, "{:.3f}".format(std_val[3]),
                    (self.roi_image.bayer_w, self.roi_image.bayer_h + h_shift + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        save_path = os.path.join(save_dir,
                                 "{}_{}_bayer.png".format(with_index, self.type.replace("\\", "/"))).replace("\\", "/")
        cv2.imwrite(save_path, save_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        with_index += 1

        # 2. mean
        mean_val = self.roi_image.mean(by_channel=True)
        mean_val_vis = np.round(normalization(mean_val, clip_min, clip_max, 0, 255)).astype(np.uint8)
        save_image[:h_shift] = 255
        save_image[h_shift:h_shift+self.roi_image.bayer_h, :h_shift+self.roi_image.bayer_w] = mean_val_vis[0]
        save_image[h_shift:h_shift+self.roi_image.bayer_h, self.roi_image.bayer_w:bayer_vis.shape[1]] = mean_val_vis[1]
        save_image[h_shift+self.roi_image.bayer_h:, :h_shift+self.roi_image.bayer_w] = mean_val_vis[2]
        save_image[h_shift+self.roi_image.bayer_h:, self.roi_image.bayer_w:bayer_vis.shape[1]] = mean_val_vis[3]
        label = "{:.3f}".format(mean_val[0])
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.rectangle(save_image,
                      (0, h_shift),
                      (text_width, h_shift + text_height + 10),
                      (255, 255, 255), -1)
        cv2.putText(save_image, label, (0, h_shift + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(save_image,
                      (self.roi_image.bayer_w, h_shift),
                      (self.roi_image.bayer_w + text_width, h_shift + text_height + 10),
                      (255, 255, 255), -1)
        cv2.putText(save_image, "{:.3f}".format(mean_val[1]),
                    (self.roi_image.bayer_w, h_shift + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(save_image,
                      (0, self.roi_image.bayer_h + h_shift),
                      (text_width, self.roi_image.bayer_h + h_shift + text_height + 10),
                      (255, 255, 255), -1)
        cv2.putText(save_image, "{:.3f}".format(mean_val[2]),
                    (0, self.roi_image.bayer_h + h_shift + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(save_image,
                      (self.roi_image.bayer_w, self.roi_image.bayer_h + h_shift),
                      (self.roi_image.bayer_w + text_width, self.roi_image.bayer_h + h_shift + text_height + 10),
                      (255, 255, 255), -1)
        cv2.putText(save_image, "{:.3f}".format(mean_val[3]),
                    (self.roi_image.bayer_w, self.roi_image.bayer_h + h_shift + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
        save_path = os.path.join(save_dir,
                                 "{}_{}_signal.png".format(with_index, self.type.replace("\\", "/"))).replace("\\", "/")
        cv2.imwrite(save_path, save_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        with_index += 1

        # 3 bgr
        bgr = self.roi_image.demosaic()
        save_path = os.path.join(save_dir,
                                 "{}_{}_rgb.png".format(with_index, self.type.replace("\\", "/"))).replace("\\", "/")
        cv2.imwrite(save_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        with_index += 1

        # 4 scaling bgr
        bgr_norm = np.round(normalization(bgr, bgr.min(), bgr.max(), 0, 255)).astype(np.uint8)
        save_path = os.path.join(save_dir,
                                 "{}_{}_rgb_scaling.png".format(with_index, self.type.replace("\\", "/"))).replace("\\", "/")
        cv2.imwrite(save_path, bgr_norm, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        with_index += 1

        # 5 3sigma bgr
        mean_val = bgr.mean()
        std_val = bgr.std()
        bgr_norm = np.round(normalization(bgr, mean_val - std_val, mean_val + std_val, 0, 255)).astype(np.uint8)
        save_path = os.path.join(save_dir,
                                 "{}_{}_rgb_3sigma.png".format(with_index, self.type.replace("\\", "/"))).replace("\\", "/")
        cv2.imwrite(save_path, bgr_norm, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        # with_index += 1

    def save_texture(self, save_dir, with_index=1):
        self.save_solid_color(save_dir, with_index)

    def save_grid(self, save_dir, with_index=1):

        # 1 bgr
        if isinstance(self.roi_image, RawInfo):
            bgr = self.roi_image.demosaic()
        else:
            bgr = self.roi_image.image
        save_path = os.path.join(save_dir,
                                 "{}_{}_rgb.png".format(with_index, self.type.replace("\\", "/"))).replace("\\",
                                                                                                                  "/")
        cv2.imwrite(save_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        with_index += 1

        # 2 bgr 3sigma clip
        mean_val = bgr.mean()
        std_val = bgr.std()
        bgr_norm = np.round(normalization(bgr, mean_val - std_val, mean_val + std_val, 0, 255)).astype(np.uint8)
        save_path = os.path.join(save_dir,
                                 "{}_{}_rgb_3sigma.png".format(with_index, self.type.replace("\\", "/"))).replace("\\",
                                                                                                                  "/")
        cv2.imwrite(save_path, bgr_norm, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        with_index += 1

        # 3 statistics
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        im_c = 3 if isinstance(self.roi_image, RawInfo) else self.roi_image.c
        channel_colors = ['b', 'g', 'r']

        N = len(self._pq['h_prj']) if im_c == 1 else len(self._pq['h_prj'][0])
        for i in range(im_c):
            axes[0, 0].plot(np.arange(N), self._pq['h_prj'][i],
                            color= channel_colors[i],
                            label="channel_{}".format(i))
        axes[0, 0].set_title('Horizotal Signal')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Magnitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        N = len(self._pq['v_prj']) if im_c == 1 else len(self._pq['v_prj'][0])
        for i in range(im_c):
            axes[0, 1].plot(np.arange(N), self._pq['v_prj'][i],
                            color= channel_colors[i],
                            label="channel_{}".format(i))
        axes[0, 1].set_title('Vertical Signal')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        freq = np.fft.fftshift(self._pq['h_frq'][i]['frequency'])
        for i in range(im_c):
            axes[1, 0].plot(freq, np.fft.fftshift(self._pq['h_frq'][i]['magnitude']),
                            color= channel_colors[i],
                            label="channel_{}".format(i))
        axes[1, 0].set_title('Horizental Frequency Spectrum')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Magnitude')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        freq = np.fft.fftshift(self._pq['v_frq'][i]['frequency'])
        for i in range(im_c):
            axes[1, 1].plot(freq, np.fft.fftshift(self._pq['v_frq'][i]['magnitude']),
                            color= channel_colors[i],
                            label="channel_{}".format(i))
        axes[1, 1].set_title('Vertical Frequency Spectrum')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        axes[1, 1].set_ylabel('Magnitude')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        freq = self._pq['h_frq'][i]['frequency']
        N = len(self._pq['h_prj']) if im_c == 1 else len(self._pq['h_prj'][0])
        freq[freq < 0] += N
        for i in range(im_c):
            axes[2, 0].plot(freq, self._pq['h_frq'][i]['magnitude'],
                            color= channel_colors[i],
                            label="channel_{}".format(i))
        axes[2, 0].set_title('Shift Horizontal Frequency Spectrum')
        axes[2, 0].set_xlabel('Frequency (Hz)')
        axes[2, 0].set_ylabel('Magnitude')
        axes[2, 0].legend()
        axes[2, 0].grid(True)

        freq = self._pq['v_frq'][i]['frequency']
        N = len(self._pq['v_prj']) if im_c == 1 else len(self._pq['v_prj'][0])
        freq[freq < 0] += N
        for i in range(im_c):
            axes[2, 1].plot(freq, self._pq['v_frq'][i]['magnitude'],
                            color= channel_colors[i],
                            label="channel_{}".format(i))
        axes[2, 1].set_title('Shift Vertical Frequency Spectrum')
        axes[2, 1].set_xlabel('Frequency (Hz)')
        axes[2, 1].set_ylabel('Magnitude')
        axes[2, 1].legend()
        axes[2, 1].grid(True)

        freq = self._pq['h_frq'][i]['frequency']
        N = len(self._pq['h_prj']) if im_c == 1 else len(self._pq['h_prj'][0])
        freq[freq < 0] += N
        for i in range(im_c):
            axes[3, 0].plot(freq[self._pq['h_frq'][i]['grid_frequcy_band'][0]:self._pq['h_frq'][i]['grid_frequcy_band'][1]],
                            self._pq['h_frq'][i]['magnitude'][self._pq['h_frq'][i]['grid_frequcy_band'][0]:self._pq['h_frq'][i]['grid_frequcy_band'][1]],
                            color=channel_colors[i],
                            label="channel_{}".format(i))
        axes[3, 0].set_title('Horizontal High Frequency Spectrum: {}'.format(self._pq['h_grid_energy']))
        axes[3, 0].set_xlabel('Frequency (Hz)')
        axes[3, 0].set_ylabel('Magnitude')
        axes[3, 0].legend()
        axes[3, 0].grid(True)

        freq = self._pq['v_frq'][i]['frequency']
        N = len(self._pq['v_prj']) if im_c == 1 else len(self._pq['v_prj'][0])
        freq[freq < 0] += N
        for i in range(im_c):
            axes[3, 1].plot(freq[self._pq['v_frq'][i]['grid_frequcy_band'][0]:self._pq['v_frq'][i]['grid_frequcy_band'][1]],
                            self._pq['v_frq'][i]['magnitude'][self._pq['v_frq'][i]['grid_frequcy_band'][0]:self._pq['v_frq'][i]['grid_frequcy_band'][1]],
                            color=channel_colors[i],
                            label="channel_{}".format(i))
        axes[3, 1].set_title('Vertical High Frequency Spectrum: {}'.format(self._pq['v_grid_energy']))
        axes[3, 1].set_xlabel('Frequency (Hz)')
        axes[3, 1].set_ylabel('Magnitude')
        axes[3, 1].legend()
        axes[3, 1].grid(True)
        save_path = os.path.join(save_dir,
                                 "{}_fft.png".format(with_index))
        plt.savefig(save_path)
        plt.close("all")

    def save_chroma(self, save_dir, with_index=1):
        save_path = os.path.join(save_dir,
                                 "{}_{}_rgb.png".format(with_index, self.type.replace("\\", "/"))).replace("\\",
                                                                                                           "/")
        cv2.imwrite(save_path, self.pq["src_image"], [cv2.IMWRITE_PNG_COMPRESSION, 0])
        with_index += 1

        # 2 bgr 3sigma clip
        save_path = os.path.join(save_dir,
                                 "{}_{}_color_fringing.png".format(with_index, self.type.replace("\\", "/"))).replace("\\",
                                                                                                                  "/")
        bgr = self.pq['fringing_image']
        cv2.putText(bgr, "color fringing: {:.3f}".format(self.pq['fringing_ratio']),(0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(save_path, bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    cls.save_result = save_result
    cls.save_solid_color = save_solid_color
    cls.save_texture = save_texture
    cls.save_grid = save_grid
    cls.save_chroma = save_chroma

    return cls