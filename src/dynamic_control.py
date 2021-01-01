from wodeutil.ml.config.dynamic_params import DynamicConfig
import time


def test_load():
    config: DynamicConfig = DynamicConfig()
    setup_dynamic_control(config)
    config.save_config(config, 'others/dynamic.params')
    c2 = DynamicConfig.load_config('others/dynamic.params')
    print(type(c2))
    print(vars(c2))


def test_control():
    config: DynamicConfig = DynamicConfig()
    set_config(config)
    config.save_config(config, 'others/dynamic.params')
    configpath = 'others/dynamic.params'
    start = 1
    for i in range(start, 101):
        if config.epoch_stop_point > 0 and i % config.epoch_stop_point == 0:
            print(f"now i equals epoch stop point! i ({i}) == epoch_stop_point({config.epoch_stop_point}")
            continue
        config = update_dynamic_params(i, dynamic_config=config, params_path=configpath)
        if print_train_cond(i, dynamic_config=config):
            print("this is i: ", i)
            print(
                f"dynamic_chk_every: {config.dynamic_chk_every} | print_every: {config.print_every} | log_train_on: {config.log_train_on}")
            print(f"epoch_stop_point: {config.epoch_stop_point} | train_print_points: {config.train_print_points}")
        if neptune_chkpoint_save_cond(i, config):
            print('neptune save chkpoint True.')
        time.sleep(2)
        #
        # if config.log_train_on:
        #     print("log_train_on is True.")


def set_config(config: DynamicConfig):
    config.epoch_stop_point = 10
    config.print_every = 2
    config.dynamic_chk_every = 2
    config.log_train_on = True
    config.train_print_points = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    config.neptune_chkpoint_every = 2
    config.neptune_save_chkpoint_on = True
    config.neptune_metrics_every = 1
    config.neptune_metrics_log_on = True


def chk_dynamic_params(step, dynamic_config):
    """
    check if need to reload the dynamic params file to update the dynamic config
    :param step:
    :param config:
    :return:
    """
    if dynamic_config and step:
        return step % dynamic_config.dynamic_chk_every == 0
    return False


def print_train_cond(step, dynamic_config: DynamicConfig):
    if step and dynamic_config:
        return dynamic_config.log_train_on and step % dynamic_config.print_every == 0
    return False


def log_neptune_metric_cond(step, dynamic_config: DynamicConfig):
    if step and dynamic_config and dynamic_config.neptune_metrics_every:
        return dynamic_config.neptune_metrics_log_on and step % dynamic_config.neptune_metrics_every == 0
    return False


def neptune_chkpoint_save_cond(step, dynamic_config: DynamicConfig):
    if step and dynamic_config and dynamic_config.neptune_save_chkpoint_on:  # return true if save_chkpoint_very is met or step is specified in save_chkpoint_points
        return step % dynamic_config.neptune_chkpoint_every == 0 or (
                dynamic_config.neptune_save_chkpoint_steps and step in dynamic_config.neptune_save_chkpoint_steps)
    return False


def chk_dynamic_config_update(step, dynamic_config: DynamicConfig):
    return step and dynamic_config and step % dynamic_config.dynamic_chk_every == 0


def update_dynamic_params(step, dynamic_config: DynamicConfig, params_path=None):
    if params_path and chk_dynamic_config_update(step, dynamic_config):
        dynamic_config = dynamic_config.load_config(params_path)
    return dynamic_config


def chk_save_cond(step, report_every, dynamic_config: DynamicConfig):
    if step and dynamic_config:  # return true if save_chkpoint_very is met or step is specified in save_chkpoint_points
        return step % dynamic_config.save_chkpoint_every == 0 or (
                dynamic_config.save_chkpoint_points and step in dynamic_config.save_chkpoint_points)
    return False


def setup_dynamic_control(config: DynamicConfig):
    # these are currently used
    config.dynamic_chk_every = 2000
    config.log_time_point = 4000
    config.neptune_metrics_every: int = 300
    config.save_chkpoint_every: int = 4000  # save checkpoint
    config.print_every: int = 1000
    config.log_train_on: bool = True

    # not used
    config.save_chkpoint_points: list = None
    config.neptune_metrics_log_points: list = None
    config.print_points: list = None
    config.log_eval_on: bool = False
    config.log_test_on: bool = False
    config.train_print_points = None
    config.eval_print_points = None
    config.test_print_points = None
    config.epoch_stop_point = -1  # negative number indicates no stop point


if __name__ == '__main__':
    test_load()
    # test_control()
