%% Code for generative model of the dark room problem

% Supplementary Code for: Active inference models do not contradict folk psychology

% By: Ryan Smith, Maxwell J. D. Ramstead, and Alex Kiefer

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% First, you need to add SPM12, and the DEM toolbox of SPM12 to your path in Matlab.

clear all
close all      % These commands clear the workspace and close any figures
rng('shuffle') % This sets the random number generator to produce different 
               % random sequences each time (you can alse set to 'default'
               % to produce same random sequence each time)

% Simulation options after model building below:

set_dislike = 1; % Larger value = greater dislike for not getting ice cream.
                 % (keep at 1 when set_desire > 0; set to 0 when set_desire = 0)
            
set_desire = 1; % Desire parameter: larger value = more urgent desire for ice cream.
                % Try values between 0 (pure epistemic drive) and 3 (strong
                % desire)

%% 1. Set up model structure

% Number of time points or 'epochs' within a trial: T
% =========================================================================

% Here, we specify 3 time points (T), in which the agent 1) starts in the
% 'dark room' state and then can either stay still, flip a light switch, or
% go to the left or the right.

T = 3;

% Priors about initial states: D and d
% =========================================================================

%--------------------------------------------------------------------------
% Specify prior probabilities about initial states in the generative 
% process (D)
%--------------------------------------------------------------------------

% For the 'context' state factor, we can specify where the ice cream actually 
% is:

D{1} = [1 0]';  % {'ice cream on the left','ice cream on the right'}

% For the 'behavior' state factor, we can specify that the agent always
% begins a trial in the 'dark room' state (i.e., before choosing to stay, 
% flip the light switch, or go left or right):

D{2} = [1 0 0 0]'; % {'stay in dark room','light switch','go left','go right'}

%--------------------------------------------------------------------------
% Specify prior beliefs about initial states in the generative model (d)
% Note: This is optional, and will simulate learning priors over states 
% if specified.
%--------------------------------------------------------------------------

% For context beliefs, we can specify that the agent starts out believing 
% that both contexts are equally likely, but with somewhat low confidence in 
% these beliefs:

d{1} = [.5 .5]';  % {'ice cream on the left','ice cream on the right'}

% For behavior beliefs, we can specify that the agent expects with 
% certainty that it will begin a trial in the 'dark room' state:

d{2} = [1 0 0 0]'; % {'stay in dark room','light switch','go left','go right'}

% State-outcome mappings: A
% =========================================================================

%--------------------------------------------------------------------------
% Specify the probabilities of outcomes given each state in the generative 
% process (A)

% This includes one matrix per outcome modality
%--------------------------------------------------------------------------

% First we specify the mapping from states to observing where the ice cream is 
% (outcome modality 1). Here, the rows correspond to observations, the columns
% correspond to the first state factor (context), and the third dimension
% corresponds to behavior. Each column is a probability distribution
% that must sum to 1.

% We start by specifying that both contexts generate the 'dark room'
% observation across all behavior states:

Ns = [length(D{1}) length(D{2})]; % number of states in each state factor (2 and 4)

for i = 1:Ns(2) 

    A{1}(:,:,i) = [1 1; % Darkness
                   0 0; % Ice cream to the left
                   0 0];% Ice cream to the right
end

% Then we specify that the 'light switch' behavior state generates
% observations about where the ice cream is, depending on the context
% state.                

A{1}(:,:,2) = [0 0;    % Darkness
               1 0;    % Ice cream left
               0 1];   % ice cream right

% Next we specify the mapping between states and getting the ice cream or not. 
% The first two behavior states ('stay' and 'flip switch') do not generate 
% either getting or not getting ice cream in either context:

for i = 1:2

    A{2}(:,:,i) = [1 1;  % Darkness
                   0 0;  % Don't get ice cream
                   0 0]; % Get ice cream
end
           
% Choosing to go left (behavior state 3) only leads to getting ice cream if ice
% cream is on the left (left column)

A{2}(:,:,3) = [0 0;  % Darkeness        
               0 1;  % Don't get ice cream
               1 0]; % Get ice cream

% Choosing to go right (behavior state 4) only leads to getting ice cream if ice
% cream is on the right (right column)
           
A{2}(:,:,4) = [0 0;  % Darkeness        
               1 0;  % Don't get ice cream
               0 1]; % Get ice cream
           
           
% Finally, we specify an identity mapping between behavior states and
% observed behaviors, to ensure the agent knows that behaviors were carried
% out as planned. Here, each row corresponds to each behavior state.
           
for i = 1:Ns(2) 

    A{3}(i,:,i) = [1 1]; % Observed behavior

end

% Controlled transitions and transition beliefs : B{:,:,u}
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the probabilistic transitions between hidden states
% under each action (sometimes called 'control states'). 
%--------------------------------------------------------------------------

% Columns are states at time t. Rows are states at t+1.

% The agent cannot control where the ice cream is, so there is only 1 'action',
% indicating that contexts remain stable within a trial:

B{1}(:,:,1) = [1 0;  % Ice cream on the left
               0 1]; % Ice cream on the right
           
% The agent can control the behavior state, and we include 4 possible 
% actions:

% Move to (or stay in) the 'dark room' state from any other state
B{2}(:,:,1) = [1 1 1 1;  % dark room
               0 0 0 0;  % light switch
               0 0 0 0;  % go left
               0 0 0 0]; % go right
           
% Move to the 'flip switch' state from any other state
B{2}(:,:,2) = [0 0 0 0;  % dark room
               1 1 1 1;  % light switch
               0 0 0 0;  % go left
               0 0 0 0]; % go right

% Move to the 'go left' state from any other state
B{2}(:,:,3) = [0 0 0 0;  % dark room
               0 0 0 0;  % light switch
               1 1 1 1;  % go left
               0 0 0 0]; % go right

% Move to the 'go right' state from any other state
B{2}(:,:,4) = [0 0 0 0;  % dark room
               0 0 0 0;  % light switch
               0 0 0 0;  % go left
               1 1 1 1]; % go right      
           

% Preferred outcomes: C
%==========================================================================

%--------------------------------------------------------------------------
% Next, we have to specify the 'prior preferences', encoded here as log
% probabilities. 
%--------------------------------------------------------------------------

% One matrix per outcome modality. Each row is an observation, and each
% columns is a time point. Negative values indicate lower preference,
% positive values indicate a high preference. Stronger preferences promote
% risky choices (urgent need to fulfill desires) and reduced information-seeking.

% We can start by setting a 0 preference for all outcomes:

No = [size(A{1},1) size(A{2},1) size(A{3},1)]; % number of outcomes in 
                                                 % each outcome modality

C{1}      = zeros(No(1),T); % Location of ice cream
C{2}      = zeros(No(2),T); % Getting ice cream or not
C{3}      = zeros(No(3),T); % Observed behaviors

% Then we can specify a 'dislike' magnitude and 'desire' magnitude at time points 2 
% and 3:

dislike = set_dislike; % By default we set this to 1 at the top of the script

desire = set_desire;  % We set this value at the top of the script. 
                      % By default we set it to 1, but try changing its value to 
                      % see how it affects model behavior (higher values will promote
                      % immediate ice cream seeking without first turning on the light, 
                      % as described in the main text, because getting 
                      % the ice cream is urgent)

C{2}(:,:) =    [0  0   0   ;           % Darkness
                0 -dislike -dislike  ; % Don't get ice cream
                0  desire  desire];    % Get ice cream
           
            
% Allowable policies: V. 
%==========================================================================

%--------------------------------------------------------------------------
% Each policy is a sequence of actions over time that the agent can 
% consider. 
%--------------------------------------------------------------------------

% Here rows correspond to time points and should be length T-1 (here, 2 
% transitions, from time point 1 to time point 2, and time point 2 to time point 3):

Np = 6; % Number of policies
Nf = 2; % Number of state factors

V         = ones(T-1,Np,Nf);

V(:,:,1) = [1 1 1 1 1 1;
            1 1 1 1 1 1]; % Context state is not controllable

V(:,:,2) = [1 2 2 2 3 4;
            1 2 3 4 3 4];
        
% For V(:,:,2), columns left to right indicate policies allowing: 
% 1. stay in the dark room
% 2. flip the light switch and stay still
% 3. flip the light switch and go left
% 4. flip the light switch and go right
% 5. go left while it's still dark
% 6. go right while it's still dark

% Additional parameters. 
%==========================================================================

% Beta: Expected precision of expected free energy (G) over policies (a 
% positive value, with higher values indicating lower expected precision).
% Lower values make policy selection less deteriministic. For our example 
% simulations we will simply set this to its default value of 1. Updates in
% beta correspond to changes in valence in the simulations shown in Figure
% 1.

     beta = 1; % By default this is set to 1

% Alpha: An 'inverse temperature' or 'action precision' parameter that 
% controls how much randomness there is when selecting actions (e.g., how 
% often the agent might choose not to flip the light switch, even if the model 
% assigned the highest probability to that action. This is a positive 
% number, where higher values indicate less randomness. Here we set this to 
% a high value:

    alpha = 32;  % Any positive number. 1 is very low, 32 is fairly high; 
                 % an extremely high value can be used to specify
                 % deterministic action (e.g., 512)

                        
%% 2. Define MDP Structure
%==========================================================================
%==========================================================================

mdp.T = T;                    % Number of time steps
mdp.V = V;                    % allowable (deep) policies
mdp.A = A;                    % state-outcome mapping
mdp.B = B;                    % transition probabilities
mdp.C = C;                    % preferred states
mdp.D = D;                    % true initial states

mdp.d = d;                    % belief about initial states
    
mdp.alpha = alpha;               % high action precision (minimal randomness in choice)
mdp.beta = beta;                 % expected precision of expected free energy over policies
                                 % (this is related to valence in the paper)

% We can add labels to states, outcomes, and actions for subsequent plotting:

label.factor{1}   = 'Door Location';   label.name{1}    = {'Left','Right'};
label.factor{2}   = 'Location';     label.name{2}    = {'Dark Room','Light Switch','Go Left','Go Right'};
label.modality{1} = 'Information';    label.outcome{1} = {'Dark Room','Left','Right'};
label.modality{2} = 'Desires';  label.outcome{2} = {'Dark Room','No Ice Cream','Ice Cream'};
label.modality{3} = 'Observed Action';  label.outcome{3} = {'Dark Room','Flip Switch','Go Left','Go Right'};
label.action{2} = {'Stay Still','Flip Switch','Go Left','Go Right'};
mdp.label = label;

clear beta
clear alpha
clear eta
clear desire
clear dislike % We clear these so we can re-specify them in later simulations

%--------------------------------------------------------------------------
% Use a script to check if all matrix-dimensions are correct:
%--------------------------------------------------------------------------
mdp = spm_MDP_check(mdp);

%% 3. Single trial simulations
 
%--------------------------------------------------------------------------
% Now that the generative process and model have been specified, we can
% simulate a single trial using the spm_MDP_VB_X script. 
%--------------------------------------------------------------------------

MDP = spm_MDP_VB_X(mdp);

% We can then use standard plotting routines to visualize simulated 
% beliefs and behavior:

spm_figure('GetWin','Figure 1'); clf    % display behavior
spm_MDP_VB_trial(MDP,1:2,1:2); 

% Please see the main text for figure interpretations

