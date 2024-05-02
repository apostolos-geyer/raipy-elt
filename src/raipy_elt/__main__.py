

def raw_to_bronze():
    import os
    from pathlib import Path

    if not (data_dir := os.getenv('DATA_DIR', None)):
        print('DATA_DIR environment variable not set. Please set it to the path of the data directory'
                         '\n(The directory that contains the subdirectory 0-raw, etc...)')
        
        print('i.e, call the script like this: DATA_DIR=/path/to/data raw2bronze')
        exit(1)

    from raipy_elt.extract.raw_to_bronze import (
        gen_stages, 
        run_stages
    )

    from raipy_elt.extract.configs import (
        TABLES, RAW
    )

    for table in TABLES[RAW]:
        stages = gen_stages(table, Path(data_dir))
        run_stages(stages)
    