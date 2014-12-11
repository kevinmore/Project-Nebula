#include "AnimatorController.h"

AnimatorController::AnimatorController( QSharedPointer<ModelManager> manager )
	: m_modelManager(manager),
	  m_actor(new GameObject)
{
	initContorlPanel();

	buildStateMachine();
	m_timer.start();
	m_actor->setMovingBehaviour(GameObject::DISCRETE);
}


AnimatorController::~AnimatorController()
{}

void AnimatorController::setCurrentClip( const QString& clipName )
{
	if (clipName != m_currentClip)
	{
		m_currentClip = clipName;
		emit currentClipChanged(m_currentClip);
	}
}

void AnimatorController::buildStateMachine()
{
	// build the state machine
	m_stateMachine = new QStateMachine();
	m_stateMachine->setObjectName("Character State Machine");

	// walking running system
	QState* moving_system = new QState(m_stateMachine);
	moving_system->setObjectName("Moving System");

	// 3 basic states
	QState* idle = createLoopingState("Idle", "m_idle", moving_system);
	idle->assignProperty(m_controlPanel->moveButton, "text", "Idle");
	idle->assignProperty(m_controlPanel->fastMoveButton, "text", "Idle");
	connect(idle, SIGNAL(entered()), m_actor, SLOT(resetSpeed()));
	m_StateClipMap[idle] = "m_idle";

	QState* walk = createLoopingState("Walk", "m_walk", moving_system);
	walk->assignProperty(m_controlPanel->moveButton, "text", "Walking");
	walk->assignProperty(m_controlPanel->turnLeftButton, "enabled", true);
	walk->assignProperty(m_controlPanel->turnRightButton, "enabled", true);
	walk->assignProperty(m_controlPanel->turnRoundButton, "enabled", true);
	m_StateClipMap[walk] = "m_walk";

	QState* run = createLoopingState("Run", "m_run", moving_system);
	run->assignProperty(m_controlPanel->moveButton, "text", "Running");
	run->assignProperty(m_controlPanel->fastMoveButton, "text", "Running");
	m_StateClipMap[run] = "m_run";

	// 9 smooth transition states
	QState* idle_to_walk = createTransitionState("Start Walking", "Starting", "m_walk_start", walk, moving_system);
	idle_to_walk->assignProperty(m_controlPanel->moveButton, "text", "Idle => Walk");
	m_StateClipMap[idle_to_walk] = "m_walk_start";

	QState* walk_to_idle = createTransitionState("Stop Walking", "Stopping", "m_walk_stop", idle, moving_system);
	walk_to_idle->assignProperty(m_controlPanel->moveButton, "text", "Walk => Idle");
	m_StateClipMap[walk_to_idle] = "m_walk_stop";

	QState* idle_to_run  = createTransitionState("Start Running", "Starting", "m_run_start", run, moving_system);
	idle_to_run->assignProperty(m_controlPanel->fastMoveButton, "text", "Idle => Run");
	m_StateClipMap[idle_to_run] = "m_run_start";

	QState* run_to_idle  = createTransitionState("Stop Running", "Stopping", "m_run_stop", idle, moving_system);
	run_to_idle->assignProperty(m_controlPanel->fastMoveButton, "text", "Run => Idle");
	m_StateClipMap[run_to_idle] = "m_run_stop";

	QState* walk_to_run  = createTransitionState("Speed Up", "Accelerating", "m_walk_to_run", run, moving_system);
	walk_to_run->assignProperty(m_controlPanel->moveButton, "text", "Walk => Run");
	m_StateClipMap[walk_to_run] = "m_walk_to_run";

	QState* run_to_walk  = createTransitionState("Slow Down", "Decelerating", "m_run_to_walk", walk, moving_system);
	run_to_walk->assignProperty(m_controlPanel->moveButton, "text", "Run => Walk");
	m_StateClipMap[run_to_walk] = "m_run_to_walk";

	QState* turnLeft_to_walk = createTransitionState("Turn Left", "Turning", "m_turn_left_60_to_walk", walk, moving_system);
	turnLeft_to_walk->assignProperty(m_controlPanel->turnLeftButton, "enabled", false);
	m_StateClipMap[turnLeft_to_walk] = "m_turn_left_60_to_walk";

	QState* turnRight_to_walk = createTransitionState("Turn Right", "Turning", "m_turn_right_60_to_walk", walk, moving_system);
	turnRight_to_walk->assignProperty(m_controlPanel->turnRightButton, "enabled", false);
	m_StateClipMap[turnRight_to_walk] = "m_turn_right_60_to_walk";

	QState* turnRound_to_walk = createTransitionState("Turn Round", "Turning", "m_turn_left_180_to_walk", walk, moving_system);
	turnRound_to_walk->assignProperty(m_controlPanel->turnRoundButton, "enabled", false);
	m_StateClipMap[turnRound_to_walk] = "m_turn_left_180_to_walk";

	// synchronize the above states to its corresponding character movement
	// moving
	syncMovement(TRANSLATION, idle);
	syncMovement(TRANSLATION, walk);
	syncMovement(TRANSLATION, run);
  	
 	//transition states
	syncMovement(TRANSLATION, idle_to_walk);
	syncMovement(TRANSLATION, walk_to_idle);

	syncMovement(TRANSLATION, idle_to_run);
	syncMovement(TRANSLATION, run_to_idle);

	syncMovement(TRANSLATION, walk_to_run);
	syncMovement(TRANSLATION, run_to_walk);

	// turning
 	syncMovement(ALL, turnLeft_to_walk, "Y, 60");
 	syncMovement(ALL, turnRight_to_walk, "Y, -60");
 	syncMovement(ALL, turnRound_to_walk, "Y, 180");

	moving_system->setInitialState(idle);

	// transition triggers
	// the normal movement button
	// idle -> walk -> run -> walk -> idle
	QEventTransition *idle_to_walkTrans = new QEventTransition(m_controlPanel->moveButton, QEvent::Enter, idle);
	idle_to_walkTrans->setObjectName("Idle to Walk");
	idle_to_walkTrans->setTargetState(idle_to_walk);

	QEventTransition *walk_to_runTrans = new QEventTransition(m_controlPanel->moveButton, QEvent::MouseButtonPress, walk);
	walk_to_runTrans->setObjectName("Walk to Run");
	walk_to_runTrans->setTargetState(walk_to_run);
	
	QEventTransition *run_to_walkTrans = new QEventTransition(m_controlPanel->moveButton, QEvent::MouseButtonRelease, run);
	run_to_walkTrans->setObjectName("Run to Walk");
	run_to_walkTrans->setTargetState(run_to_walk);

	QEventTransition *walk_to_idleTrans = new QEventTransition(m_controlPanel->moveButton, QEvent::Leave, walk);
	walk_to_idleTrans->setObjectName("Walk to Idle");
	walk_to_idleTrans->setTargetState(walk_to_idle);

	// the fast movement button
	// idle -> run -> idle
	QEventTransition *idle_to_runTrans = new QEventTransition(m_controlPanel->fastMoveButton, QEvent::MouseButtonPress, idle);
	idle_to_runTrans->setObjectName("Idle to Run");
	idle_to_runTrans->setTargetState(idle_to_run);

	QEventTransition *run_to_idleTrans = new QEventTransition(m_controlPanel->fastMoveButton, QEvent::MouseButtonRelease, run);
	run_to_idleTrans->setObjectName("Run to Idle");
	run_to_idleTrans->setTargetState(run_to_idle);

	// the turning buttons
	// idle - > turn -> walk
	QEventTransition* idle_to_turnLeft = new QEventTransition(m_controlPanel->turnLeftButton, QEvent::MouseButtonPress, idle);
	idle_to_turnLeft->setObjectName("Turn Left");
	idle_to_turnLeft->setTargetState(turnLeft_to_walk);

	QEventTransition* idle_to_turnRight = new QEventTransition(m_controlPanel->turnRightButton, QEvent::MouseButtonPress, idle);
	idle_to_turnRight->setObjectName("Turn Right");
	idle_to_turnRight->setTargetState(turnRight_to_walk);

	QEventTransition* idle_to_turnRound = new QEventTransition(m_controlPanel->turnRoundButton, QEvent::MouseButtonPress, idle);
	idle_to_turnRound->setObjectName("Turn Around");
	idle_to_turnRound->setTargetState(turnRound_to_walk);


	// social system
	QState* social_system = new QState(m_stateMachine);
	social_system->setObjectName("Social System");

	QState* talk = new QState(social_system);
	talk->setObjectName("Talk");

	QState* listen = new QState(social_system);
	listen->setObjectName("Listen");

	
	// start the state machine
	m_stateMachine->setInitialState(moving_system);
	m_stateMachine->start();
}

QState* AnimatorController::createBasicState( const QString& stateName, const QString& clipName, QState* parent /*= 0*/ )
{
	RiggedModel* man = m_modelManager->getRiggedModel(clipName);

	QState* pState = new QState(parent);

	pState->setObjectName(stateName);
	pState->assignProperty(this, "currentClip", clipName);

	// entering this state should restart the timer in order to play the animation from the beginning
	connect(pState, SIGNAL(entered()), this, SLOT(restartTimer()));

	return pState;
}


QState* AnimatorController::createLoopingState( const QString& stateName, const QString& clipName, QState* parent /*= 0*/ )
{
	RiggedModel* man = m_modelManager->getRiggedModel(clipName);

	QState* pState = new QState(parent);
	pState->setObjectName(stateName);
	pState->assignProperty(this, "currentClip", clipName);

	QTimer *timer = new QTimer(pState);
	timer->setInterval((int)(man->animationDuration() * 1000 - 32)); // minus 2 frame duration to keep the animation synchronized
	timer->setSingleShot(true);
	QState *timing = new QState(pState);
	timing->setObjectName("Looping");
	connect(timing, SIGNAL(entered()), timer, SLOT(start()));
	timing->addTransition(timer, SIGNAL(timeout()), pState);

	pState->setInitialState(timing);

	// entering this state should restart the timer in order to play the animation from the beginning
	connect(timing, SIGNAL(entered()), this, SLOT(restartTimer()));
	connect(pState, SIGNAL(exited()), this, SLOT(finishingLoopingState()));

	return pState;
}

QState* AnimatorController::createTransitionState( const QString& stateName, const QString& subStateName, const QString& clipName, 
												QState* doneState, QState* parent /*= 0*/ )
{
	RiggedModel* man = m_modelManager->getRiggedModel(clipName);

	QState* pState = new QState(parent);
	
	pState->setObjectName(stateName);
	pState->assignProperty(this, "currentClip", clipName);

	QTimer *timer = new QTimer(pState);
	timer->setInterval((int)(man->animationDuration() * 1000 - 32)); // minus 2 frame duration to keep the animation synchronized
	timer->setSingleShot(true);
	QState *timing = new QState(pState);
	timing->setObjectName(subStateName);
	connect(timing, SIGNAL(entered()), timer, SLOT(start()));
	timing->addTransition(timer, SIGNAL(timeout()), doneState);

	pState->setInitialState(timing);

	// entering this state should restart the timer in order to play the animation from the beginning
	connect(timing, SIGNAL(entered()), this, SLOT(restartTimer()));

	return pState;
}


void AnimatorController::syncMovement( SYNC_OPTION option, QState* pState, const QString& customData /*= ""*/)
{
	QString clipName = m_StateClipMap[pState];
	RiggedModel* man = m_modelManager->getRiggedModel(clipName);
	QString paramString;
	const char* slot;
	if (option == TRANSLATION)
	{
		slot = SLOT(translateInWorld(const QString&));
		vec3 delta = man->getRootTranslation();
		paramString = QString::number((float)delta.x()) + ", " + 
					  QString::number((float)delta.y()) + ", " + 
					  QString::number((float)delta.z());
	}
	else if (option == ROTATION)
	{
// 		slot = SLOT(rotateInWorld(const QString&));
// 		QQuaternion delta = man->getRootRotation();
// 		paramString = QString::number((float)delta.scalar()) + ", " + 
// 					  QString::number((float)delta.x()) + ", " + 
// 					  QString::number((float)delta.y()) + ", " + 
// 					  QString::number((float)delta.z());
		slot = SLOT(rotateInWorldAxisAndAngle(const QString&));
		paramString = customData;
	}
	else if (option == ALL)
	{
		syncMovement(TRANSLATION, pState);
		syncMovement(ROTATION, pState, customData);
		return;
	}
	
	QSignalMapper *mapper = new QSignalMapper(pState);
	connect(pState, SIGNAL(exited()), mapper, SLOT(map()));
	mapper->setMapping(pState, paramString);
	connect(mapper, SIGNAL(mapped(const QString&)), m_actor, slot);
}


void AnimatorController::render(const float globalTime)
{
	RiggedModel* man = m_modelManager->getRiggedModel(m_currentClip);
	float time = (float)m_timer.elapsed()/1000;

	man->setActor(m_actor);
	man->render(time);
}

void AnimatorController::restartTimer()
{
	m_timer.restart();
}

void AnimatorController::initContorlPanel()
{
	m_controlPanel = new AnimatorPanel();
	m_controlPanel->show();
}

void AnimatorController::stateMachineFinised()
{
	setCurrentClip("m_idle");
}

void AnimatorController::finishingLoopingState()
{
	float time = (float)m_timer.elapsed()/1000;
	RiggedModel* man = m_modelManager->getRiggedModel(m_currentClip);
	float duration = man->animationDuration();
	// if the animation stops before the time duration for it
	// synchronize the game object position
	if (duration - time > 0.01f)
	{
		// the game object need to move back words
		vec3 delta = - man->getRootTranslation() * (1 - time / duration);
		man->getActor()->translateInWorld(delta);
	}
}
