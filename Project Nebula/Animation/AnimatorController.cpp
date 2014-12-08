#include "AnimatorController.h"

AnimatorController::AnimatorController( QSharedPointer<ModelManager> manager, QObject* handelingWidget )
	: m_modelManager(manager),
	  m_handler(handelingWidget),
	  m_actor(new GameObject())
{
	buildStateMachine();
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
	QState* idle = createBasicState("Idle", "m_idle", moving_system);
	QState* walk = createBasicState("Walk", "m_walk", moving_system);
	QState* run = createBasicState("Run", "m_run", moving_system);

	// 9 smooth transition states
	QState* idle_to_walk = createTimedSubState("Start Walking", "Starting", "m_walk_start", walk, moving_system);
	QState* walk_to_idle = createTimedSubState("Stop Walking", "Stopping", "m_walk_stop", idle, moving_system);

	QState* idle_to_run  = createTimedSubState("Start Running", "Starting", "m_run_start", run, moving_system);
	QState* run_to_idle  = createTimedSubState("Stop Running", "Stopping", "m_run_stop", idle, moving_system);

	QState* walk_to_run  = createTimedSubState("Speed Up", "Accelerating", "m_walk_to_run", run, moving_system);
	QState* run_to_walk  = createTimedSubState("Slow Down", "Decelerating", "m_run_to_walk", walk, moving_system);

	QState* turnLeft_to_walk = createTimedSubState("Turn Left", "Turning", "m_turn_left_60_to_walk", walk, moving_system);
	QState* turnRight_to_walk = createTimedSubState("Turn Right", "Turning", "m_turn_right_60_to_walk", walk, moving_system);
	QState* turnRound_to_walk = createTimedSubState("Turn Round", "Turning", "m_turn_left_180_to_walk", walk, moving_system);

	// connect the above states to its corresponding character movement
	// moving
	syncMovement(idle, SIGNAL(entered()), SLOT(setSpeed(const QString&)), "0, 0, 0");
	syncMovement(walk, SIGNAL(entered()), SLOT(setSpeed(const QString&)), "0, 0, 100");
	syncMovement(run,  SIGNAL(entered()), SLOT(setSpeed(const QString&)), "0, 0, 200");
	
	//transition states
	syncMovement(idle_to_walk, SIGNAL(entered()), SLOT(setSpeed(const QString&)), "0, 0, 0");
	syncMovement(walk_to_idle, SIGNAL(entered()), SLOT(setSpeed(const QString&)), "0, 0, 0");
	syncMovement(idle_to_run,  SIGNAL(entered()), SLOT(setSpeed(const QString&)), "0, 0, 0");
	syncMovement(run_to_idle,  SIGNAL(entered()), SLOT(setSpeed(const QString&)), "0, 0, 0");
	syncMovement(walk_to_run,  SIGNAL(entered()), SLOT(setSpeed(const QString&)), "0, 0, 0");
	syncMovement(run_to_walk,  SIGNAL(entered()), SLOT(setSpeed(const QString&)), "0, 0, 0");

	// turning
	syncMovement(turnLeft_to_walk,  SIGNAL(exited()), SLOT(rotate(const QString&)), "Y, 60");
	syncMovement(turnRight_to_walk, SIGNAL(exited()), SLOT(rotate(const QString&)), "Y, -60");
	syncMovement(turnRound_to_walk, SIGNAL(exited()), SLOT(rotate(const QString&)), "Y, 180");


	moving_system->setInitialState(idle);

	// transitions
	QKeyEventTransition* idle_to_turnLeft = new QKeyEventTransition(m_handler, QEvent::KeyPress, Qt::Key_A, idle);
	idle_to_turnLeft->setObjectName("Press A");
	idle_to_turnLeft->setTargetState(turnLeft_to_walk);

	QKeyEventTransition* idle_to_turnRight = new QKeyEventTransition(m_handler, QEvent::KeyPress, Qt::Key_D, idle);
	idle_to_turnRight->setObjectName("Press D");
	idle_to_turnRight->setTargetState(turnRight_to_walk);

	QKeyEventTransition* idle_to_turnRound = new QKeyEventTransition(m_handler, QEvent::KeyPress, Qt::Key_S, idle);
	idle_to_turnRound->setObjectName("Press S");
	idle_to_turnRound->setTargetState(turnRound_to_walk);


	QKeyEventTransition* idle_to_walkTrans = new QKeyEventTransition(m_handler, QEvent::KeyPress, Qt::Key_W, idle);
	idle_to_walkTrans->setObjectName("Press W");
	idle_to_walkTrans->setTargetState(idle_to_walk);

	QKeyEventTransition* walk_to_idleTrans = new QKeyEventTransition(m_handler, QEvent::KeyRelease, Qt::Key_W, walk);
	walk_to_idleTrans->setObjectName("Release W");
	walk_to_idleTrans->setTargetState(walk_to_idle);

	QKeyEventTransition* walk_to_runTrans = new QKeyEventTransition(m_handler, QEvent::KeyPress, Qt::Key_Shift, walk);
	walk_to_runTrans->setObjectName("Press Shift");
	walk_to_runTrans->setTargetState(walk_to_run);

	QKeyEventTransition* run_to_walkTrans = new QKeyEventTransition(m_handler, QEvent::KeyRelease, Qt::Key_Shift, run);
	run_to_walkTrans->setObjectName("Release Shift");
	run_to_walkTrans->setTargetState(run_to_walk);

	QKeyEventTransition* idle_to_runTrans = new QKeyEventTransition(m_handler, QEvent::KeyPress, Qt::Key_W, idle);
	idle_to_runTrans->setModifierMask(Qt::ShiftModifier);
	idle_to_runTrans->setObjectName("Press Shift + W");
	idle_to_runTrans->setTargetState(idle_to_run);

	QKeyEventTransition* run_to_idleTrans = new QKeyEventTransition(m_handler, QEvent::KeyRelease, Qt::Key_W, run);
	run_to_idleTrans->setObjectName("Release W");
	run_to_idleTrans->setTargetState(run_to_idle);

	// social system
	QState* social_system = new QState(m_stateMachine);
	social_system->setObjectName("Social System");

	QState* talk = new QState(social_system);
	talk->setObjectName("Talk");

	QState* listen = new QState(social_system);
	listen->setObjectName("Listen");

	// entering each state should restart the timer in order to play the animation from the beginning
	for (int i = 0; i < moving_system->children().count(); ++i)
	{
		QState* pState = (QState*)moving_system->children()[i];
		connect(pState, SIGNAL(entered()), m_handler, SLOT(restartTimer()));
	}

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
	pState->assignProperty(this, "duration", man->animationDuration());

	return pState;
}



QState* AnimatorController::createTimedSubState( const QString& stateName, const QString& subStateName, const QString& clipName, 
												QState* doneState, QState* parent /*= 0*/ )
{
	RiggedModel* man = m_modelManager->getRiggedModel(clipName);

	QState* pState = new QState(parent);
	
	pState->setObjectName(stateName);
	pState->assignProperty(this, "currentClip", clipName);
	pState->assignProperty(this, "duration", man->animationDuration());

	QTimer *timer = new QTimer(pState);
	timer->setInterval((int)man->animationDuration() * 1000);
	timer->setSingleShot(true);
	QState *timing = new QState(pState);
	timing->setObjectName(subStateName);
	connect(timing, SIGNAL(entered()), timer, SLOT(start()));
	timing->addTransition(timer, SIGNAL(timeout()), doneState);
	pState->setInitialState(timing);

	return pState;
}


void AnimatorController::syncMovement( QState* pState, const char* signal, const char* slot, const QString& paramString )
{
	QSignalMapper *mapper = new QSignalMapper(this);
	connect(pState, signal, mapper, SLOT(map()));
	mapper->setMapping(pState, paramString);
	connect(mapper, SIGNAL(mapped(const QString&)), m_actor, slot);
}
