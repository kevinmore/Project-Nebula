#pragma once
#include "AnimatorController.h"

class NPCController : public AnimatorController
{
	Q_OBJECT
public:
	NPCController(QSharedPointer<ObjectManager> manager, const QString& defualtClip, const QString& responding);
	~NPCController();

	void setDefaultClip(const QString& clipName) { m_defaultClip = clipName; }
	inline const QString& defaultClip() { return m_defaultClip; }

	void setRespondingClip(const QString& clipName) { m_respondingClip = clipName; }
	inline const QString& respondingClip() { return m_respondingClip; }

public slots:
	void listenToEvents(const QString& event);

signals:
	void reactToEvents();

private:
	void buildStateMachine();
	QString m_defaultClip, m_respondingClip;
};

