#pragma once
#include <Scene/RiggedModel.h>
#include <Scene/Managers/ModelManager.h>
#include <QStateMachine>
#include <QElapsedTimer>
#include "AnimatorPanel.h"

class AnimatorController : public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString currentClip READ currentClip WRITE setCurrentClip)

public:
	AnimatorController(QSharedPointer<ModelManager> manager);
	~AnimatorController();

	GameObject* getActor() { return m_actor; }

	QStateMachine* getStateMachine() { return m_stateMachine; }

	QString currentClip() const { return  m_currentClip; }

	void render();

signals:
	void currentClipChanged(const QString& clipName);

public slots:
	void setCurrentClip(const QString& clipName);
	void restartTimer();
	void stateMachineFinised();
	void finishingLoopingState();

private:
	QSharedPointer<ModelManager> m_modelManager;
	QStateMachine* m_stateMachine;

	QString m_currentClip;
	float m_durationInSeconds;
	QElapsedTimer m_timer;

	GameObject* m_actor;

	AnimatorPanel* m_controlPanel;

	QMap<QState*, QString> m_StateClipMap;

	enum SYNC_OPTION
	{
		TRANSLATION,
		ROTATION,
		ALL
	};

	void buildStateMachine();
	QState* createBasicState(const QString& stateName, const QString& clipName, QState* parent = 0);
	QState* createLoopingState(const QString& stateName, const QString& clipName, QState* parent = 0);
	QState* createTransitionState(const QString& stateName, const QString& subStateName, const QString& clipName, 
								QState* doneState, QState* parent = 0);

	void syncMovement(SYNC_OPTION option, QState* pState, const QString& customData = "");

	void initContorlPanel();
};

