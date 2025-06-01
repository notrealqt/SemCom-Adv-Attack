import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses
from .SwinJSCC import SwinJSCC # Import the SwinJSCC model

class JSCC_CNN(object):
    def __init__(self, k, n, seed=None, filename=None, learning_rate=0.001,
                 swin_embed_dim=128, swin_encoder_depths=[2], swin_encoder_num_heads=[4],
                 swin_decoder_depths=[2], swin_decoder_num_heads=[4],
                 swin_window_size=1, swin_mlp_ratio=4., swin_qkv_bias=True,
                 swin_qk_scale=None, swin_drop_rate=0., swin_attn_drop_rate=0.,
                 swin_drop_path_rate=0.1):
        self.k = k 
        self.n = n
        self.bits_per_symbol = self.k / self.n
        self.M = 2**self.k
        self.seed = seed
        if self.seed is not None:
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)

        self.optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
        
        # Build the Swin JSCC model using the imported class
        self.model = SwinJSCC(
            k=self.k, n=self.n, M=self.M,
            embed_dim=swin_embed_dim,
            encoder_depths=swin_encoder_depths, encoder_num_heads=swin_encoder_num_heads,
            decoder_depths=swin_decoder_depths, decoder_num_heads=swin_decoder_num_heads,
            window_size=swin_window_size,
            mlp_ratio=swin_mlp_ratio,
            qkv_bias=swin_qkv_bias,
            qk_scale=swin_qk_scale,
            drop_rate=swin_drop_rate,
            attn_drop_rate=swin_attn_drop_rate,
            drop_path_rate=swin_drop_path_rate
        )

        # For saving/loading weights
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.checkpoint_manager = None
        if filename: 
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=filename, max_to_keep=5)
            if self.checkpoint_manager.latest_checkpoint:
                self.load(self.checkpoint_manager.latest_checkpoint)
                print(f"Restored from {self.checkpoint_manager.latest_checkpoint}")
            else:
                print(f"Initializing from scratch (no checkpoint found in {filename}).")
        else:
            print("Initializing from scratch (no filename provided).")

    def _build_swin_jscc_model(self):
        # This method is now effectively replaced by instantiating SwinJSCC directly.
        # Kept for conceptual structure, but the actual model is self.model from __init__.
        # If we need to re-build or get sub-parts, we use self.model.get_encoder(), etc.
        pass # Model is built in __init__

    def EbNo2Sigma(self, ebnodb):
        '''Convert Eb/No in dB to noise standard deviation'''
        ebno = 10**(ebnodb / 10.0)
        # Assuming average symbol energy is 1 due to normalization in encoder
        # For JSCC, this might need re-evaluation based on how 'bits_per_symbol' applies
        # If k/n is still relevant (e.g. k source bits to n complex channel symbols)
        return 1.0 / np.sqrt(2.0 * self.bits_per_symbol * ebno + 1e-8)

    @tf.function
    def train_step(self, s_batch, p_batch, noise_std_val, learning_rate_val):
        '''A single training step'''
        with tf.GradientTape() as tape:
            s_hat_logits = self.model([s_batch, noise_std_val, p_batch], training=True)
            loss = self.loss_fn(s_batch, s_hat_logits)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.learning_rate = learning_rate_val
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Performance metrics
        predictions = tf.argmax(s_hat_logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, s_batch), tf.float32))
        bler = 1.0 - accuracy
        return loss, bler

    @tf.function
    def test_step(self, s_batch, p_batch, noise_std_val):
        '''Compute the BLER over a single batch'''
        s_hat_logits = self.model([s_batch, noise_std_val, p_batch], training=False)
        
        predictions = tf.argmax(s_hat_logits, axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, s_batch), tf.float32))
        bler = 1.0 - accuracy
        return bler, s_hat_logits

    def train(self, p_perturbation, training_params, validation_params, dr_out_rate=0.0):
        '''Training and validation loop'''
        p_perturbation_tensor = tf.constant(p_perturbation, dtype=tf.float32)

        for epoch_params_idx, params in enumerate(training_params):            
            batch_size, lr, ebnodb, iterations = params            
            print(f"Epoch Set {epoch_params_idx+1}: Batch Size: {batch_size}, LR: {lr}, EbNodB: {ebnodb}, Iterations: {iterations}")
            
            val_batch_size, val_ebnodb, val_steps = validation_params[epoch_params_idx]
            noise_std_train = self.EbNo2Sigma(ebnodb)
            noise_std_val = self.EbNo2Sigma(val_ebnodb)

            for i in range(iterations):
                # Generate a batch of random messages
                s_batch_train = tf.random.uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
                
                loss, bler_train = self.train_step(
                    s_batch_train, 
                    p_perturbation_tensor[:batch_size] if p_perturbation_tensor.shape[0] >= batch_size else tf.tile(p_perturbation_tensor, [batch_size // p_perturbation_tensor.shape[0] + 1, 1, 1])[:batch_size], # Ensure p_batch matches batch_size
                    tf.constant(noise_std_train, dtype=tf.float32),
                    tf.constant(lr, dtype=tf.float32)
                    # tf.constant(True, dtype=tf.bool) # is_training
                )    
                
                if (i + 1) % val_steps == 0:
                    s_batch_val = tf.random.uniform(shape=[val_batch_size], minval=0, maxval=self.M, dtype=tf.int64)
                    bler_val, _ = self.test_step(
                        s_batch_val, 
                        p_perturbation_tensor[:val_batch_size] if p_perturbation_tensor.shape[0] >= val_batch_size else tf.tile(p_perturbation_tensor, [val_batch_size // p_perturbation_tensor.shape[0] + 1, 1, 1])[:val_batch_size],
                        tf.constant(noise_std_val, dtype=tf.float32)
                        # tf.constant(False, dtype=tf.bool) # is_training
                    )
                    print(f"Iteration {i+1}/{iterations}: Train Loss: {loss.numpy():.4f}, Train BLER: {bler_train.numpy():.4f}, Val BLER: {bler_val.numpy():.4f}")
        return       

    def save(self, checkpoint_dir_path):
        '''Save the current model'''
        if self.checkpoint_manager is None:
             self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=checkpoint_dir_path, max_to_keep=5)
        save_path = self.checkpoint_manager.save()
        print(f"Model saved to {save_path}")
        return save_path
    
    def load(self, checkpoint_path_or_dir):
        '''Load a pre-trained model from a specific checkpoint file or the latest in a directory'''
        if tf.io.gfile.isdir(checkpoint_path_or_dir):
            if self.checkpoint_manager is None:
                self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=checkpoint_path_or_dir, max_to_keep=5)
            status = self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
            if self.checkpoint_manager.latest_checkpoint:
                 print(f"Restored from latest checkpoint: {self.checkpoint_manager.latest_checkpoint}")
            else:
                print(f"No checkpoint found in directory: {checkpoint_path_or_dir}")
        else: # Assuming it's a file
            status = self.checkpoint.restore(checkpoint_path_or_dir)
            print(f"Restored from file: {checkpoint_path_or_dir}")
        status.expect_partial() # Use expect_partial if optimizer state or other parts might not be loaded, e.g. when loading only weights.
        return status

    def transmit(self, s_messages):
        '''Returns the transmitted signals corresponding to message indices'''
        s_tensor = tf.constant(s_messages, dtype=tf.int64)
        encoder_submodel = self.model.get_encoder() # Use the method from SwinJSCC
        return encoder_submodel(s_tensor, training=False).numpy()

    # --- Attack and Simulation Methods (Adapted from AE_CNN, may need further Swin JSCC specific adjustments) ---
    
    def bler_sim_attack_AWGN(self, p_perturbation, PSR_dB, ebnodbs, batch_size, iterations, dr_out_rate=0.0):
        '''Generate the BLER for different cases.
           Note: This is a simplified adaptation. Swin JSCC might behave differently under these attacks.
        '''
        PSR = 10**(PSR_dB / 10.0)
        # Ensure p_perturbation is a numpy array for norm calculation
        p_perturbation_np = np.array(p_perturbation)
        scale_factor = np.sqrt((PSR * self.n) / (np.linalg.norm(p_perturbation_np)**2 + 1e-8))
        p_adv_np = scale_factor * p_perturbation_np
        p_adv_tensor = tf.constant(p_adv_np, dtype=tf.float32)
        
        BLER_no_attack = np.zeros_like(ebnodbs, dtype=float)
        BLER_adv_attack = np.zeros_like(ebnodbs, dtype=float)
        BLER_jamming = np.zeros_like(ebnodbs, dtype=float)
        
        p_zero_tensor = tf.zeros_like(p_adv_tensor, dtype=tf.float32)

        for _ in range(iterations):
            s_batch = tf.random.uniform(shape=[batch_size], minval=0, maxval=self.M, dtype=tf.int64)
            
            # No attack
            for idx, ebnodb in enumerate(ebnodbs):
                noise_std = self.EbNo2Sigma(ebnodb)
                bler, _ = self.test_step(s_batch, p_zero_tensor, tf.constant(noise_std, dtype=tf.float32))
                BLER_no_attack[idx] += bler.numpy()
            
            # Adversarial attack
            for idx, ebnodb in enumerate(ebnodbs):
                noise_std = self.EbNo2Sigma(ebnodb)
                bler, _ = self.test_step(s_batch, p_adv_tensor, tf.constant(noise_std, dtype=tf.float32))
                BLER_adv_attack[idx] += bler.numpy()
            
            # Jamming attack
            normal_noise_as_jammer_np = np.random.normal(0, 1, p_adv_np.shape)
            jamming_np = np.linalg.norm(p_adv_np) * (normal_noise_as_jammer_np / (np.linalg.norm(normal_noise_as_jammer_np) + 1e-8))
            jamming_tensor = tf.constant(jamming_np, dtype=tf.float32)
            for idx, ebnodb in enumerate(ebnodbs):
                noise_std = self.EbNo2Sigma(ebnodb)
                bler, _ = self.test_step(s_batch, jamming_tensor, tf.constant(noise_std, dtype=tf.float32))
                BLER_jamming[idx] += bler.numpy()
                
        return BLER_no_attack / iterations, BLER_adv_attack / iterations, BLER_jamming / iterations

    def fgm_attack(self, s_target_message_idx, p_initial_perturbation, ebnodb, epsilon_scale=0.1):
        ''' Placeholder for FGM attack adapted for Swin JSCC.
            This needs careful implementation of gradient calculation through the Swin model.
        '''
        print("Warning: fgm_attack is a placeholder and needs Swin JSCC specific implementation.")
        s_target_tensor = tf.constant([s_target_message_idx], dtype=tf.int64)
        p_tensor = tf.Variable(p_initial_perturbation, dtype=tf.float32)
        noise_std_val = self.EbNo2Sigma(ebnodb)

        with tf.GradientTape() as tape:
            tape.watch(p_tensor)
            # Note: The model takes a batch. Here s_target_tensor is a single message.
            # We might need to generate a dummy batch or adapt.
            # For simplicity, assuming a batch of 1 for the target message.
            s_dummy_batch = tf.repeat(s_target_tensor, repeats=p_tensor.shape[0] if p_tensor.ndim == 3 else 1, axis=0)

            s_hat_logits = self.model([s_dummy_batch, 
                                       tf.constant(noise_std_val, dtype=tf.float32), 
                                       p_tensor], training=False)
            # Loss against the original message (if trying to cause misclassification)
            # or against a target class if that's the goal.
            # Assuming s_target_message_idx is the TRUE label we want to perturb away from.
            loss = self.loss_fn(s_dummy_batch, s_hat_logits) 
        
        gradient = tape.gradient(loss, p_tensor)
        if gradient is None:
            return np.zeros_like(p_initial_perturbation), s_target_message_idx, 0.0

        perturbation_update = tf.sign(gradient) * epsilon_scale # Simple FGM step
        
        # This is a very simplified FGM, the original AE_CNN fgm_attack is more complex.
        return perturbation_update.numpy(), s_target_message_idx, epsilon_scale # Placeholder return

    def UAPattack_fgm(self, ebnodb, num_samples, PSR_dB, epsilon_scale=0.1):
        ''' Placeholder for UAP attack adapted for Swin JSCC.
            This also needs careful Swin JSCC specific implementation.
        '''
        print("Warning: UAPattack_fgm is a placeholder and needs Swin JSCC specific implementation.")
        universal_perturbation_np = np.zeros((1, 2, self.n), dtype=np.float32)
        
        PSR_val = 10**(PSR_dB / 10.0)
        
        for i in range(num_samples):
            s_sample_idx = np.random.randint(0, self.M)
            s_sample_tensor = tf.constant([s_sample_idx], dtype=tf.int64)
            
            # Check if current UAP causes misclassification
            current_uap_tensor = tf.constant(universal_perturbation_np, dtype=tf.float32)
            bler, s_hat_logits = self.test_step(s_sample_tensor, current_uap_tensor, self.EbNo2Sigma(ebnodb))
            
            predicted_label = tf.argmax(s_hat_logits, axis=1)[0].numpy()

            if predicted_label == s_sample_idx: # If still correctly classified, try to find perturbation
                # Use a simplified FGM step to find a perturbation for this sample
                # This is a conceptual placeholder
                temp_p_var = tf.Variable(current_uap_tensor, dtype=tf.float32)
                with tf.GradientTape() as tape:
                    tape.watch(temp_p_var)
                    s_hat_logits_uap = self.model([s_sample_tensor, self.EbNo2Sigma(ebnodb), temp_p_var], training=False)
                    loss_uap = self.loss_fn(s_sample_tensor, s_hat_logits_uap)
                
                grad_uap = tape.gradient(loss_uap, temp_p_var)
                if grad_uap is not None:
                    delta_perturbation_np = (tf.sign(grad_uap) * epsilon_scale).numpy()
                    universal_perturbation_np += delta_perturbation_np[0] # Assuming delta_perturbation is [1,2,n]

                    # Project UAP to satisfy PSR constraint
                    norm_uap = np.linalg.norm(universal_perturbation_np)
                    max_norm = np.sqrt(PSR_val * self.n / (2*self.n)) # Simplified constraint, may need adjustment
                                                                    # The original AE_CNN uses Epsilon_uni based on norm(UAP)
                    
                    if norm_uap > max_norm and norm_uap > 1e-8:
                         universal_perturbation_np = universal_perturbation_np * (max_norm / norm_uap)
            if (i+1)%10 == 0:
                print(f"UAP generation: Sample {i+1}/{num_samples}")

        return universal_perturbation_np

    def weight_variable(self, shape): # This is more of a TF1.x utility. In TF2.x, Keras layers handle initialization.
        '''Xavier-initialized weights. For Keras layers, use kernel_initializer='glorot_uniform'.'''
        # This method is kept for structural similarity but Keras layers handle this.
        # If used for tf.Variable directly:
        initializer = tf.initializers.GlorotUniform()
        return tf.Variable(initializer(shape=shape, dtype=tf.float32))

# Example Usage (Illustrative)
if __name__ == '__main__':
    k_param = 4
    n_param = 7
    
    # Instantiate JSCC_CNN which internally creates SwinJSCC
    jscc_system = JSCC_CNN(
        k=k_param, n=n_param, seed=42, filename=None, # No checkpoint loading for this example
        swin_embed_dim=64, # Smaller for quick test
        swin_encoder_depths=[1], swin_encoder_num_heads=[2],
        swin_decoder_depths=[1], swin_decoder_num_heads=[2],
        swin_window_size=1, 
        swin_mlp_ratio=2.0
    )

    print("JSCC_CNN system initialized with SwinJSCC model.")
    jscc_system.model.summary(line_length=120)

    # Dummy training parameters for a very short run
    dummy_training_params = [
        # batch_size, lr, ebnodb, iterations
        (32, 0.001, 10.0, 5) 
    ]
    dummy_validation_params = [
        # val_batch_size, val_ebnodb, val_steps (frequency of validation)
        (32, 10.0, 2)
    ]
    dummy_perturbation = np.zeros((1, 2, n_param), dtype=np.float32) # Single perturbation, will be tiled

    print("\nStarting dummy training...")
    jscc_system.train(dummy_perturbation, dummy_training_params, dummy_validation_params)
    print("Dummy training finished.")

    # Test transmit method
    test_messages = np.random.randint(0, jscc_system.M, size=(5,))
    transmitted_signals = jscc_system.transmit(test_messages)
    print(f"\nTransmitted signals for messages {test_messages}:\nShape: {transmitted_signals.shape}")

    # Test BLER simulation (very short)
    ebnodbs_test = np.array([5.0, 10.0])
    print("\nRunning BLER simulation (AWGN)...")
    bler_no_attack, bler_adv, bler_jam = jscc_system.bler_sim_attack_AWGN(
        p_perturbation=dummy_perturbation, 
        PSR_dB=0, 
        ebnodbs=ebnodbs_test, 
        batch_size=16, 
        iterations=2 # Very few iterations for quick test
    )
    print(f"BLER No Attack: {bler_no_attack}")
    print(f"BLER Adv Attack: {bler_adv}")
    print(f"BLER Jamming: {bler_jam}")

    # Placeholder for FGM attack test (would require more setup)
    # print("\nTesting FGM (placeholder)...")
    # p_fgm, _, _ = jscc_system.fgm_attack(s_target_message_idx=0, p_initial_perturbation=dummy_perturbation, ebnodb=7.0)
    # print(f"FGM perturbation (shape): {p_fgm.shape}")

    print("\nExample run completed.")
