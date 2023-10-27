for ds in  fever vc # scifact
do
    for shot in 3 6 12 24 48
    do
        for seed in 0 2 3 4
        do
            for st in 1500 
            do
                if [[ ${shot} -eq 3 ]]
                then
                    r=$((${st}/2))
                    g=1
                elif [[ ${shot} -eq 6 ]]
                then
                    r=$((${st}/2))
                    g=2
                else
                    r=$((${st}/(${shot}/4)))
                    g=2
                fi
                python -m src.pl_train -c t03b.json+lora.json -k  exp_name=t03b_${ds}_shot${shot}_seed${seed} few_shot_random_seed=${seed} seed=${seed} dataset=${ds} batch_size=1 grad_accum_factor=1 num_steps=${st} eval_batch_size=4 num_shot=${shot} eval_epoch_interval=${r}
            done
        done
    done
done
