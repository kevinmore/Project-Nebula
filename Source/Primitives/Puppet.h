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

	Puppet(GameObjectPtr go, Variable val, float amount, float duration = 0.0f);

private slots:
	void update();

private:
	GameObjectPtr m_target;
	Variable m_variable;
	float m_amount;
	float m_duration;
	float m_updateRate;
};

