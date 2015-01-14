#pragma once
#include <Scene/Managers/ObjectManager.h>
#include <QStateMachine>
#include <QElapsedTimer>
#include "AnimatorPanel.h"

class NPCController;
class AnimatorController : public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString currentClip READ currentClip WRITE setCurrentClip)

public:
	AnimatorController(QSharedPointer<ObjectManager> manager);
	~AnimatorController();

	inline GameObject* getActor() { return m_actor; }

	inline QStateMachine* getStateMachine() { return m_stateMachine; }

	inline QString currentClip() const { return  m_currentClip; }

	void render(const float globalTime);

	void addSocialTargets(NPCController* target) { m_socialTargets << target; }
	inline QVector<NPCController*> getSocialTargets() { return m_socialTargets; }
	virtual void buildStateMachine();

signals:
	void currentClipChanged(const QString& clipName);

public slots:
	void setCurrentClip(const QString& clipName);
	void restartTimer();
	void stateMachineFinised();
	void finishingLoopingState();

protected:
	QSharedPointer<ObjectManager> m_modelManager;
	QStateMachine* m_stateMachine;

	QString m_currentClip;
	float m_durationInSeconds;
	QElapsedTimer m_timer;

	GameObject* m_actor;

	AnimatorPanel* m_controlPanel;

	QMap<QState*, QString> m_StateClipMap;

	QVector<NPCController*> m_socialTargets;

	enum SYNC_OPTION
	{
		TRANSLATION,
		ROTATION,
		ALL
	};

	QState* createBasicState(const QString& stateName, const QString& clipName, QState* parent = 0);
	QState* createLoopingState(const QString& stateName, const QString& clipName, QState* parent = 0);
	QState* createTransitionState(const QString& stateName, const QString& subStateName, const QString& clipName, 
								QState* doneState, QState* parent = 0);

	void syncMovement(SYNC_OPTION option, QState* pState, const QString& customData = "");

	void initContorlPanel();
	void mappingConnection(QObject *sender, const char *signal, const QString& paramString, QObject *receiver, const char *slot);
};

