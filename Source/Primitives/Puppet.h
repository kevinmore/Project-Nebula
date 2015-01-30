#pragma once
#include <Primitives/GameObject.h>

class Puppet : public QObject
{
	Q_OBJECT

public:

	enum Variable
	{
		Position,
		Rotation,
		Scale,
		Color
	};

	Puppet(GameObject* go, Variable val, const vec3& speed, float duration = 0.0f);

private slots:
	void update();
	void destroy();

private:
	GameObject* m_target;
	Variable m_variable;
	vec3 m_speed;
	float m_duration;
	float m_updateRate;
};

typedef QSharedPointer<Puppet> PuppetPtr;
