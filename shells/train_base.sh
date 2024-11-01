export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export OMP_NUM_THREADS=4

agents=(
    # ippo
    # mappo
    # tarmac
    maaf_rnn
    # maaf_attn
)

envs=(
    # mpe.simple_spread_v3
    # mpe.simple_tag_v3
    mpe.tag_v15
    # smac.3m
    # smac.8m
)

num_layers=3
hidden_dim=128 
message_num_layers=3
message_hidden_dim=128
activation=Tanh
message_activation=Tanh 
message_dim=48

lr=1e-4

# seed=5
env_max_cycles=100
num_steps=128
total_timesteps=2000000

training_timesteps=$total_timesteps

for seed in 0 1 2 
do 
for agent in ${agents[@]} 
do 
    if [ $agent == "maaf_rnn" ]; then
        # Divide the parameter by 2
        training_timesteps=$(( $total_timesteps / 2 ))
    fi
    if [ $agent == "maaf_attn" ]; then
        # Divide the parameter by 2
        training_timesteps=$(( $total_timesteps / 2 ))
    fi

    echo $agent
    echo $training_timesteps

    #$ ---------------------------------------
    for update_epochs in 8
    do
    for clip_coef in 0.3 # 0.2 0.3 0.4
    do 
    for env_id in ${envs[@]}
    do 
    if [[ "$env_id" == *"smac"* ]]; then
        num_envs=1
    else
        num_envs=8
    fi

    exp_name='paper/base_hyperparams'
    msg=$msg_activation'_'$message_dim
    base=$update_epochs'_'$clip_coef
    save_dir='outputs/'$exp_name'/'$env_id'/'$agent'/'$base'/seed_'$seed

    python tasks/train_base.py \
        --env-id $env_id \
        --num-envs $num_envs \
        --agent $agent \
        --clip-coef $clip_coef \
        --save-dir $save_dir \
        --num-steps $num_steps \
        --update-epochs $update_epochs \
        --num-layers $num_layers \
        --total-timesteps $training_timesteps \
        --hidden-dim $hidden_dim \
        --activation $activation \
        --learning-rate $lr \
        --env-max-cycles $env_max_cycles \
        --seed $seed \
        --message-dim $message_dim \
        --message-hidden-dim $message_hidden_dim \
        --message-num-layers $message_num_layers \
        --message-activation $message_activation
    done 
    done 
    done 
    done  
done 