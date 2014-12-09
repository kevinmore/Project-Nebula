#include "AnimatorController.h"

AnimatorController::AnimatorController( QSharedPointer<ModelManager> manager, QObject* handelingWidget )
	: m_modelManager(manager),
	  m_handler(handelingWidget),
	  m_actor(new GameObject())
{
	buildStateMachine();
	m_timer.start();
	// restart the timer whenever an animation is finished
	connect(this, SIGNAL(animationCycleDone()), this, SLOT(restartTimer()));
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
	//QState* idle = createBasicState("Idle", "m_idle", moving_system);
	QState* walk = createBasicState("Walk", "m_walk", moving_system);
	QState* run = createBasicState("Run", "m_run", moving_system);
	QState* idle = createTimedSubState("Idle", "Looping", "m_idle", nullptr, moving_system);

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
	syncMovement(TRANSLATION, "m_walk");
	syncMovement(TRANSLATION, "m_run");
 	
 	//transition states
	syncMovement(TRANSLATION, "m_walk_start");
	syncMovement(TRANSLATION, "m_walk_stop");

	syncMovement(TRANSLATION, "m_run_start");
	syncMovement(TRANSLATION, "m_run_stop");

	syncMovement(TRANSLATION, "m_walk_to_run");
	syncMovement(TRANSLATION, "m_run_to_walk");

	// turning
// 	syncMovement(ALL, turnLeft_to_walk, "m_turn_left_60_to_walk", SIGNAL(exited()));
// 	syncMovement(ALL, turnRight_to_walk, "m_turn_right_60_to_walk", SIGNAL(exited()));
// 	syncMovement(ROTATION, turnRound_to_walk, "m_turn_left_180_to_walk", SIGNAL(exited()));

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
		connect(pState, SIGNAL(entered()), this, SLOT(restartTimer()));
	}

	// start the state machine
	m_stateMachine->setInitialState(moving_system);
	m_stateMachine->start();
}

QState* AnimatorController::createBasicState( const QString& stateName, const QString& clipName, QState* parent /*= 0*/ )
{
	RiggedModel* man = m_modelManager->getRiggedModel(clipName);
	vec3 averageSpeed = man->getRootTranslation() / man->animationDuration();

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
	vec3 averageSpeed = man->getRootTranslation() / man->animationDuration();

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


void AnimatorController::syncMovement( MOVEMENT_TYPE type, const QString& clipName )
{
	RiggedModel* man = m_modelManager->getRiggedModel(clipName);
	QString paramString;
	const char* slot;
	if (type == TRANSLATION)
	{
		slot = SLOT(translateInWorld(const QString&));
		vec3 delta = man->getRootTranslation();
		paramString = QString::number((float)delta.x()) + ", " + 
					  QString::number((float)delta.y()) + ", " + 
					  QString::number((float)delta.z());
	}
	else if (type == ROTATION)
	{
		slot = SLOT(rotateInWorld(const QString&));
		QQuaternion delta = man->getRootRotation();
		paramString = QString::number((float)delta.scalar()) + ", " + 
					  QString::number((float)delta.x()) + ", " + 
					  QString::number((float)delta.y()) + ", " + 
					  QString::number((float)delta.z());
	}
	else if (type == ALL)
	{
		syncMovement(TRANSLATION, clipName);
		syncMovement(ROTATION, clipName);
		return;
	}
	
	QSignalMapper *mapper = new QSignalMapper(this);
	connect(this, SIGNAL(animationCycleDone()), mapper, SLOT(map()));
	mapper->setMapping(this, paramString);
	connect(mapper, SIGNAL(mapped(const QString&)), m_actor, slot);
}

void AnimatorController::render()
{
	RiggedModel* man = m_modelManager->getRiggedModel(m_currentClip);
	float time = (float)m_timer.elapsed()/1000;
	man->setActor(m_actor);
	man->render(time);
	if (man->animationDuration() - time < 0.01f)
	{
		emit animationCycleDone();
		qDebug() << "render cycle done!";
		// add logic here, animation done
	}
}

void AnimatorController::restartTimer()
{
	m_timer.restart();
}
