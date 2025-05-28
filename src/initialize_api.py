import logging
import logging.config
import os

import yaml

def logger_configure(config_path: str = "./config/logging.yaml"):
    with open(config_path, "rt", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Проходим по всем хэндлерам и создаём директории, если есть filename
    for handler in cfg.get("handlers", {}).values():
        filename = handler.get("filename")
        if filename:
            log_dir = os.path.dirname(filename) or "."
            os.makedirs(log_dir, exist_ok=True)

    # Наконец, применяем конфигурацию
    logging.config.dictConfig(cfg)

    

if __name__ == "__main__":
    logger_configure()
    logger = logging.getLogger('app')
    logger.info('sdsa')
    logger.warning('sdsa')
    logger.error('s93423-')
    logger.debug('4554456')