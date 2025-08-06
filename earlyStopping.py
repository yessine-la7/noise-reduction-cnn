import numpy as np
from trainLogging import setup_logging
import logging


logger = logging.getLogger()

class EarlyStopping:
    """
    Stops training when the validation loss stops improving.
    Adaptive min_delta: Auto-adjust from observed std of initial val_losses.
    """
    def __init__(self, patience=5, min_delta=None, verbose=False, adapt_start_epoch=5):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.adapt_start_epoch = adapt_start_epoch
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.val_history = []

    def __call__(self, val_loss):
        self.val_history.append(val_loss)

        if self.min_delta is None and len(self.val_history) > self.adapt_start_epoch:
            recent = np.array(self.val_history[-self.adapt_start_epoch:])
            std = recent.std()
            self.min_delta = float(std) if std > 0 else 1e-4
            if self.verbose:
                logger.info(f"[EarlyStopping] Adaptive min_delta gesetzt = {self.min_delta:.6f}")

        if self.best_loss is None:
            self.best_loss = val_loss
            if self.verbose:
                logger.info(f"[EarlyStopping] Initial best_loss = {val_loss:.4f}")
        elif val_loss < self.best_loss - (self.min_delta or 0):
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                logger.info(f"[EarlyStopping] Verbesserung: best_loss = {val_loss:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.warning(f"[EarlyStopping] Keine Verbesserung für {self.counter} Epochen")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.error(f"[EarlyStopping] Stoppe nach {self.patience} Epochen ohne Verbesserung.")
        return self.early_stop
















# class EarlyStopping:
#     """
#     Stops training when the validation loss stops improving.
#     """
#     def __init__(self, patience=5, min_delta=0.0, verbose=False):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.verbose = verbose
#         self.best_loss = None
#         self.counter = 0
#         self.early_stop = False

#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#             if self.verbose:
#                 print(f"[EarlyStopping] Initial best_loss = {val_loss:.4f}")
#         elif val_loss < self.best_loss - self.min_delta:
#             self.best_loss = val_loss
#             self.counter = 0
#             if self.verbose:
#                 print(f"[EarlyStopping] Improvement: best_loss = {val_loss:.4f}")
#         else:
#             self.counter += 1
#             if self.verbose:
#                 print(f"[EarlyStopping] No improvement for {self.counter} epochs")
#             if self.counter >= self.patience:
#                 self.early_stop = True
#                 if self.verbose:
#                     print(f"[EarlyStopping] Triggered after {self.patience} epochs without improvement.")
#         return self.early_stop
