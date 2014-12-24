#include "NPCController.h"

NPCController::NPCController(QSharedPointer<ModelManager> manager, const QString& defualtClip, const QString& responding)
	: AnimatorController(manager),
	  m_defaultClip(defualtClip),
	  m_respondingClip(responding)
{
	buildStateMachine();
}


NPCController::~NPCController()
{
}

void NPCController::listenToEvents( const QString& event )
{

	emit reactToEvents();
}

void NPCController::buildStateMachine()
{
	// build the state machine
	m_stateMachine = new QStateMachine();
	m_stateMachine->setObjectName("NPC State Machine");

	QState* default = createLoopingState("Default State", m_defaultClip, m_stateMachine);
	m_StateClipMap[default] = m_defaultClip;

	QState* response = createTransitionState("Response", "Responding", m_respondingClip, default, m_stateMachine);
	m_StateClipMap[response] = m_respondingClip;

	default->addTransition(this, SIGNAL(reactToEvents()), response);

	m_stateMachine->setInitialState(default);
	m_stateMachine->start();
}
