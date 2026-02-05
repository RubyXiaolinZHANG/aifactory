import sys
import time
import random
import numpy as np
import cv2, io
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from matplotlib import pyplot as plt

import trackio
from loguru import logger
from tqdm import tqdm
from aifactory.utils.yaml_printer import YAMLTreePrinter


class MyLog:

    _raw_log = False

    def set_raw_log(self, enable):
        self._raw_log = enable

    def raw_log_enable(self):
        self._raw_log = True

    def raw_log_disable(self):
        self._raw_log = False


class TqdmLog(MyLog):

    def __init__(self):
        pass

    def info(self, message):
        tqdm.write(message)

    def warning(self, message):
        self.info(message)


class ExperimentLogger(MyLog):
    """
    A logger that encapsulates Loguru (for operational logs) and TrackIO (for experiment metrics).
    Suitable for standardized logging in AI model training, compression, pruning, and other experiments.
    """

    def __init__(self,
                 project_name: str,
                 experiment_name: str,
                 config: Dict[str, Any],
                 log_dir: str = "./logs",
                 use_trackio: bool = True,
                 space_id: str | None = None,
                 resume: str = "never"):
        """
        Initializes the experiment logger.

        Args:
            project_name (str): TrackIO project name.
            experiment_name (str): Name of the experiment, suggested to include key parameters for distinction.
            config (Dict[str, Any]): Dictionary of experiment hyperparameters.
            log_dir (str): Storage directory for Loguru log files.
            use_trackio (bool): Whether to enable TrackIO for logging experiment metrics.
            trackio_offline (bool): Whether TrackIO runs in offline mode (data stored locally only).
        """

        self.project_name = project_name
        self.experiment_name = experiment_name
        self.config = config
        self.use_trackio = use_trackio
        self.current_epoch = None

        # 1. Initialize and configure Loguru (for operational logs)
        self._setup_loguru(log_dir)
        logger.info(f"Experiment '{experiment_name}' logging system initialized.")

        # 2. Initialize TrackIO (for experiment metrics)
        self.trackio_exp = None
        if use_trackio:
            try:
                if resume == "never":
                    experiment_name = "{}_{}".format(experiment_name, datetime.now().strftime('%Y%m%d_%H%M'))
                self.trackio_exp = trackio.init(
                    project=project_name,
                    name=experiment_name,
                    config=config,
                    space_id=space_id
                )
                logger.info(f"TrackIO experiment initialized. Experiment Name: {self.trackio_exp.name}")
                logger.info(f"TrackIO online: {space_id}")
            except Exception as e:
                logger.error(f"Failed to initialize TrackIO: {e}. Will continue using Loguru only.")
                self.use_trackio = False
        else:
            logger.warning("TrackIO is disabled. Only operational logs will be recorded.")

        # 3. print config
        self.info("{}\n".format("=" * 100), raw=True)
        self.info("{}\n".format("CONFIG"), raw=True)
        self.info("{}\n".format("=" * 100), raw=True)
        yaml_printer = YAMLTreePrinter(self.config, self)
        yaml_printer.print_tree()
        self.info("\n", raw=True)
        self.info("{}\n".format("=" * 100), raw=True)
        self.info("{}\n".format("CONFIG END!"), raw=True)
        self.info("{}\n".format("=" * 100), raw=True)

    def _setup_loguru(self, log_dir: str):
        """Configures Loguru to output to console and file."""
        # Create the log directory
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Remove Loguru's default handler
        logger.remove()

        # Add console output (concise format for real-time monitoring)
        logger.add(
            sink=sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <level>{level:8}</level> | <cyan>{module}</cyan> - <level>{message}</level>",
            colorize=True
        )

        # Add file output (detailed format for archiving and debugging)
        log_file = log_path / f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logger.add(
            sink=log_file,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",  # Rotate log file when it exceeds 10 MB
            retention="30 days"  # Keep logs for 30 days
        )

    def set_epoch(self, epoch: int):
        """Sets the current training epoch for subsequent logging convenience."""
        self.current_epoch = epoch
        logger.debug(f"Current epoch set to: {epoch}")

    # ==================== Operational Logging Methods (Loguru) ====================
    def info(self, message: str, raw=None):
        """Logs a general informational message."""
        if raw is None:
            raw = self._raw_log
        if raw:
            logger.opt(raw=True).info(message)
        else:
            logger.info(message)

    def debug(self, message: str):
        """Logs a debug message."""
        logger.debug(message)

    def warning(self, message: str):
        """Logs a warning message."""
        logger.warning(message)

    def error(self, message: str):
        """Logs an error message."""
        logger.error(message)

    def critical(self, message: str):
        """Logs a critical error message."""
        logger.critical(message)

    # ==================== Experiment Metrics Logging Methods (TrackIO) ====================
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Logs experiment metrics to TrackIO.

        Args:
            metrics (Dict[str, Any]): Dictionary of metrics, e.g., {'train/loss': 0.5, 'accuracy': 0.9}.
            step (Optional[int]): Current step number (e.g., epoch). If None, uses self.current_epoch.
        """
        if not self.use_trackio or self.trackio_exp is None:
            # Even without TrackIO, log key metrics via Loguru
            logger.debug(f"[Metrics Log] {metrics}")
            return

        try:
            # Use the provided step, current epoch, or default to None (TrackIO auto-increments)
            actual_step = step if step is not None else self.current_epoch
            self.trackio_exp.log(metrics, step=actual_step)
            logger.debug(f"Metrics logged to TrackIO (step={actual_step}): {metrics}")
        except Exception as e:
            logger.error(f"Error logging metrics to TrackIO: {e}")

    def log_image(self, image, caption="", step=None, tag="image"):
        """
        Log an image to Trackio.
        Args:
            image: Can be a numpy array, PIL.Image object, or image file path.
            caption (str): Title or description for the image.
            step (int, optional): Associated training step or epoch.
        """
        # Wrap the image into a wandb.Image object
        trackio_image = trackio.Image(image, caption=caption)
        # Log to Trackio
        self.trackio_exp.log({f"{tag}/{caption}": trackio_image}, step=step)
        self.debug(f"Image logged: {caption}")

    def log_multiple_images(self, images_dict, caption_prefix="", step=None, tag="batch"):
        """
        在单一步骤中记录多张图像到Trackio。
        Args:
            images_dict (dict): 一个字典，键为图像标识，值为图像数据（数组、路径等）。
                               例如：{"input": img1_array, "output": img2_array}
            step (int, optional): 记录步数。
            caption_prefix (str): 可选的标题前缀，用于区分不同步骤。
        """
        logged_images = {}
        for key, image_data in images_dict.items():
            caption = f"{caption_prefix}_{key}" if caption_prefix else key
            # 包装为 wandb.Image
            logged_images[f"{tag}/{key}"] = trackio.Image(image_data, caption=caption)

        # 一次调用记录所有图像
        self.trackio_exp.log(logged_images, step=step)
        self.debug(f"Multiple images logged at step {step}: {list(images_dict.keys())}")

    def log_image_sequence(self, images, key, caption_prefix="", step=None):
        """
        在不同步骤中，用相同的键记录图像以形成序列。
        Args:
            image_data: 单张图像数据。
            key (str): 用于序列记录的固定键名。
            step (int): 必须提供，代表序列中的时间步。
            caption (str): 图像标题。
        """
        if isinstance(images, list):
            for image_id, image in enumerate(images):
                wandb_image = trackio.Image(image, caption="{}{}".format(caption_prefix, "_{}".format(image_id)))
                self.trackio_exp.log({key: wandb_image}, step=image_id if step is None else step + image_id)
        elif isinstance(images, dict):
            for image_id, (image_name, image) in enumerate(images.items()):
                wandb_image = trackio.Image(image, caption="{}{}".format(caption_prefix,image_name))
                self.trackio_exp.log({key: wandb_image}, step=image_id if step is None else step + image_id)

        self.debug(f"Image added to sequence '{key}' at step {step}")

    def log_heatmap(self, matrix, caption="Heatmap", xlabel="", ylabel="", step=None):
        """
        Generate and log a heatmap to Trackio, using OpenCV for image conversion.
        Args:
            matrix (np.ndarray): 2D array, the data used to generate the heatmap.
            caption (str): Title for the heatmap.
            xlabel/ylabel (str): Axis labels.
            step (int, optional): Associated training step or epoch.
        """
        # Use matplotlib to create the heatmap
        fig, ax = plt.subplots()
        im = ax.imshow(matrix, cmap='viridis', aspect='auto')
        ax.set_title(caption)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.colorbar(im, ax=ax)
        fig.tight_layout()

        # --- 核心修复：使用OpenCV转换Figure ---
        # 1. 将图形保存到一个内存缓冲区（BytesIO对象），格式为PNG
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)

        # 2. 将缓冲区数据转换为numpy数组（一维的字节流）
        buf_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        # 3. 使用cv2.imdecode解码字节流，得到图像数组
        #    注意：cv2.imdecode 读取的是BGR格式
        cv2_img_bgr = cv2.imdecode(buf_array, cv2.IMREAD_COLOR)

        # 4. 将颜色通道从BGR转换为RGB（因为wandb.Image期望RGB）
        cv2_img_rgb = cv2.cvtColor(cv2_img_bgr, cv2.COLOR_BGR2RGB)

        # 5. 将RGB格式的numpy数组传递给wandb.Image()
        wandb_image = trackio.Image(cv2_img_rgb, caption=caption)

        # Log to Trackio
        self.trackio_exp.log({f"heatmaps/{caption}": wandb_image}, step=step)
        self.debug(f"Heatmap logged (via OpenCV): {caption}")

        # Close the figure and buffer to free memory
        plt.close(fig)
        buf.close()

    def update_config(self, new_params: Dict[str, Any]):
        """
        Updates the experiment configuration (for recording dynamically adjusted hyperparameters).
        Note: This operation's success may depend on the TrackIO backend support.
        """
        self.config.update(new_params)
        if self.use_trackio and self.trackio_exp is not None:
            try:
                # Attempt to update the configuration within TrackIO
                self.trackio_exp.config.update(new_params)
            except:
                pass
        logger.info(f"Experiment configuration updated: {new_params}")

    # ==================== Resource Cleanup ====================
    def finish(self):
        """Finishes the experiment and cleans up resources (especially TrackIO)."""
        logger.info(f"Experiment '{self.experiment_name}' is wrapping up...")
        if self.use_trackio and self.trackio_exp is not None:
            try:
                # Log a final status marker or duration
                self.trackio_exp.log({"status": "completed"}, step=self.current_epoch)
                self.trackio_exp.finish()
                logger.info("TrackIO experiment recording completed.")
            except Exception as e:
                logger.error(f"Error finalizing TrackIO experiment: {e}")
        logger.success("Experiment logger shut down.")


def main():

    experiment_config = {
        "model": "ResNet-50",
        "task": "Pruning",
        "dataset": "CIFAR-10",
        "initial_lr": 0.01,
        "batch_size": 128,
        "pruning_method": "L1Unstructured",
        "target_sparsity": 0.8,
    }

    exp_logger = ExperimentLogger(project_name="ai_factory",
                 experiment_name="test_log_epoch-5",
                 config=experiment_config,
                 log_dir="./logs",
                 use_trackio=True,
                 space_id=None)
    exp_logger.info("start testing ExperimentLogger")
    # 3. 开始你的训练/压缩流程
    total_epochs = 5
    exp_logger.info("=" * 50)
    exp_logger.info("start to simulate a training process")
    exp_logger.info("=" * 50)
    sample_idx = 10
    for epoch in range(1, total_epochs + 1):
        exp_logger.set_epoch(epoch)

        # 模拟训练
        exp_logger.debug(f"EPOCH {epoch} TRAINING ...")
        time.sleep(0.1)
        train_loss = 1.0 / epoch + random.uniform(-0.1, 0.1)
        train_acc = 0.8 + epoch * 0.05 + random.uniform(-0.02, 0.02)

        # 记录训练指标
        exp_logger.log_metrics({
            "train/loss": train_loss,
            "train/accuracy": train_acc,
            "learning_rate": 0.01 * (0.95 ** epoch)
        })
        exp_logger.info(f"EPOCH {epoch}: LOSS={train_loss:.4f}, ACC={train_acc:.4f}")

        fake_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        exp_logger.log_image(fake_image, caption="Random_Sample", step=epoch)

        images_to_log = {}
        for i in range(4):
            images_to_log[f"sample_{i}_original"] =  np.zeros( (224, 224, 3), dtype=np.uint8)
            images_to_log[f"sample_{i}_reconstructed"] =  np.zeros( (224, 224, 3), dtype=np.uint8) + 255
        exp_logger.log_multiple_images(images_to_log, step=epoch, caption_prefix=f"epoch{epoch}_compare")

        feature_map = np.zeros( (224, 224, 3), dtype=np.uint8) + 100
        exp_logger.log_image_sequence(
            image_data=feature_map,
            key=f"features/sample_{sample_idx}",  # 固定键名
            step=epoch,  # 变化的步数
            caption=f"Epoch {epoch}"
        )
        feature_map = np.zeros((224, 224, 3), dtype=np.uint8) + 100
        exp_logger.log_image_sequence(
            image_data=feature_map,
            key=f"features/sample_{sample_idx + 1}",  # 固定键名
            step=epoch,  # 变化的步数
            caption=f"Epoch {epoch}"
        )

        attention_weights = np.random.rand(3, 20, 30)
        for i in range(attention_weights.shape[0]):
            exp_logger.log_heatmap(
                matrix=attention_weights[i],
                caption="Layer_Attention_Mean/kernel-{}".format(i),
                xlabel="Head Dimension",
                ylabel="Token Position",
                step=epoch
            )

        # 模拟在特定轮次进行剪枝
        if epoch % 2 == 0:
            exp_logger.warning(f"EPOCH {epoch}: EXECUTING PRUNE  ...")
            sparsity = epoch * 0.1
            # 模拟剪枝后精度变化
            pruned_acc = train_acc - random.uniform(0.01, 0.05)

            # 记录剪枝相关指标
            exp_logger.log_metrics({
                "prune/sparsity": sparsity,
                "eval/accuracy_after_pruning": pruned_acc
            })

            # 也可以更新配置（例如记录实际达到的稀疏度）
            exp_logger.update_config({"actual_sparsity_epoch_{epoch}": sparsity})

    # 4. 实验结束，清理资源
    exp_logger.info("Training pipeline finished. Saving final model ...")
    # ... (你的模型保存代码)
    exp_logger.finish()


def test_save_frames():
    from aifactory.utils.get_files import get_target_files
    import cv2
    import os
    src_dir = "F:/database/vimeo_png/sequences/00001"
    exp_logger = ExperimentLogger(project_name="ai_factory",
                                  experiment_name="test_log_frames",
                                  config={},
                                  log_dir="./logs",
                                  use_trackio=True,
                                  space_id=None)
    folders = os.listdir(src_dir)
    count = 0
    for folder in folders:
        full_path = os.path.join(src_dir, folder)
        if not(os.path.isdir(full_path)):
            continue
        files = get_target_files(full_path, suffix=".png")
        for file_id, file in enumerate(files):
            cv_image = cv2.imread(file)
            exp_logger.log_image(cv_image, caption="sample/00001/{}".format(folder), step=file_id)
        count += 1
        if count == 5:
            break

    exp_logger.info("Test image log finished.")
    exp_logger.finish()
    return


def test_save_muli_frames():
    from aifactory.utils.get_files import get_target_files
    import cv2
    import os
    src_dir = "F:/database/vimeo_png/sequences/00001"
    exp_logger = ExperimentLogger(project_name="ai_factory",
                                  experiment_name="test_log_frames",
                                  config={},
                                  log_dir="./logs",
                                  use_trackio=True,
                                  space_id=None)
    folders = os.listdir(src_dir)
    count = 0
    for folder in folders:
        full_path = os.path.join(src_dir, folder)
        if not(os.path.isdir(full_path)):
            continue
        files = get_target_files(full_path, suffix=".png")
        image_dict = {}
        for file_id, file in enumerate(files):
            cv_image = cv2.imread(file)
            image_dict[os.path.basename(file)] = cv_image
        exp_logger.log_multiple_images(image_dict,
                                       caption_prefix="sample/00001/{}".format(folder),
                                       step=count,
                                       tag="vimeo")
        count += 1
        if count == 5:
            break

    exp_logger.info("Test image log finished.")
    exp_logger.finish()
    return


def test_save_frame_sequence():
    from aifactory.utils.get_files import get_target_files
    import cv2
    import os
    src_dir = "F:/database/vimeo_png/sequences/00001"
    exp_logger = ExperimentLogger(project_name="ai_factory",
                                  experiment_name="test_log_frames",
                                  config={},
                                  log_dir="./logs",
                                  use_trackio=True,
                                  space_id=None)
    folders = os.listdir(src_dir)
    count = 0
    for folder in folders:
        full_path = os.path.join(src_dir, folder)
        if not(os.path.isdir(full_path)):
            continue
        files = get_target_files(full_path, suffix=".png")
        image_dict = {}
        for file_id, file in enumerate(files):
            cv_image = cv2.imread(file)
            image_dict[os.path.basename(file)] = cv_image
        exp_logger.log_image_sequence(image_dict,
                                       key="train/vimeo/00001/{}".format(folder))
        count += 1
        if count == 5:
            break

    exp_logger.info("Test image log finished.")
    exp_logger.finish()
    return



if __name__ == "__main__":
    # main()
    # test_save_frames()
    # test_save_muli_frames()
    test_save_frame_sequence()