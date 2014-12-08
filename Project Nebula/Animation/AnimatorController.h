#pragma once
#include <Scene/RiggedModel.h>
#include <Scene/Managers/ModelManager.h>
#include <QStateMachine>
#include <QElapsedTimer>

class AnimatorController : public QObject
{
	Q_OBJECT
	Q_PROPERTY(QString currentClip READ currentClip WRITE setCurrentClip)
	Q_PROPERTY(float duration READ duration WRITE setDuration)

public:
	AnimatorController(QSharedPointer<ModelManager> manager, QObject* handelingWidget);
	~AnimatorController();

	GameObject* getActor() { return m_actor; }

	QStateMachine* getStateMachine() { return m_stateMachine; }

	float duration() { return m_durationInSeconds; }
	void setDuration(float dur) { m_durationInSeconds = dur; }

	QString currentClip() const { return  m_currentClip; }

	void render();

signals:
	void currentClipChanged(const QString& clipName);
	void animationCycleDone();

public slots:
	void setCurrentClip(const QString& clipName);
	void restartTimer();

private:
	QSharedPointer<ModelManager> m_modelManager;
	QObject* m_handler;
	QStateMachine* m_stateMachine;

	QString m_currentClip;
	float m_durationInSeconds;
	QElapsedTimer m_timer;

	GameObject* m_actor;

	void buildStateMachine();
	QState* createBasicState(const QString& stateName, const QString& clipName, QState* parent = 0);
	QState* createTimedSubState(const QString& stateName, const QString& subStateName, const QString& clipName, 
								QState* doneState, QState* parent = 0);

	void syncMovement(QState* pState, const char* signal, const char* slot, const QString& paramString);
};

