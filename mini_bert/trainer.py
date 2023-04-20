import torch
import animator


class BERTTrainer:
    LEARNING_RATE = 1e-3
    N_STEPS_TO_PRINT = 1000

    def __init__(self, net, train_iter, loss, vocab_size):
        self.net = net
        self.train_iter = train_iter
        self.loss = loss
        self.vocab_size = vocab_size
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.net.to(self.device)

    def train_bert(self, num_steps, use_animator=False):
        trainer = torch.optim.Adam(self.net.parameters(), lr=self.LEARNING_RATE)
        if use_animator:
            ani = animator.Animator(
                xlabel="step", ylabel="loss", xlim=[1, num_steps], legend=["mlm", "nsp"]
            )
        step = 0
        timer = animator.Timer()
        metric = animator.Accumulator(4)
        num_steps_reached = False
        while step < num_steps and not num_steps_reached:
            for (
                tokens_X,
                segments_X,
                valid_lens_x,
                pred_positions_X,
                mlm_weights_X,
                mlm_Y,
                nsp_y,
            ) in self.train_iter:
                tokens_X = tokens_X.to(self.device)
                segments_X = segments_X.to(self.device)
                valid_lens_x = valid_lens_x.to(self.device)
                pred_positions_X = pred_positions_X.to(self.device)
                mlm_weights_X = mlm_weights_X.to(self.device)
                mlm_Y = mlm_Y.to(self.device)
                nsp_y = nsp_y.to(self.device)

                trainer.zero_grad()

                timer.start()
                mlm_l, nsp_l, total_loss = self._get_batch_loss_bert(
                    tokens_X,
                    segments_X,
                    valid_lens_x,
                    pred_positions_X,
                    mlm_weights_X,
                    mlm_Y,
                    nsp_y,
                )
                total_loss.backward()
                trainer.step()
                metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
                timer.stop()

                if use_animator:
                    ani.add(step + 1, (metric[0] / metric[3], metric[1] / metric[3]))
                step += 1
                if step % self.N_STEPS_TO_PRINT == 0:
                    self._print_stats(timer, metric)
                if step == num_steps:
                    num_steps_reached = True
                    break

        self._print_stats(timer, metric)

    def _get_batch_loss_bert(
        self,
        tokens_X,
        segments_X,
        valid_lens_x,
        pred_positions_X,
        mlm_weights_X,
        mlm_Y,
        nsp_y,
    ):
        # Forward pass
        _, mlm_Y_hat, nsp_Y_hat = self.net(
            tokens_X, segments_X, valid_lens_x.reshape(-1), pred_positions_X
        )
        # Compute masked language model loss
        mlm_l = self.loss(
            mlm_Y_hat.reshape(-1, self.vocab_size), mlm_Y.reshape(-1)
        ) * mlm_weights_X.reshape(-1, 1)
        mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
        # Compute next sentence prediction loss
        nsp_l = self.loss(nsp_Y_hat, nsp_y)
        total_loss = mlm_l + nsp_l
        return mlm_l, nsp_l, total_loss

    def _print_stats(self, timer, metric):
        print(
            f"MLM loss {metric[0] / metric[3]:.3f}, "
            f"NSP loss {metric[1] / metric[3]:.3f}"
        )
        print(
            f"{metric[2] / timer.sum():.1f} sentence pairs/sec on "
            f"{str(self.device)}"
        )
