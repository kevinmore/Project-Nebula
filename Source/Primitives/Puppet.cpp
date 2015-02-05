#include "Puppet.h"
#include <QTimer>

Puppet::Puppet( GameObject* go, Variable val, const vec3& speed, float duration /*= 0.0f*/ )
	: m_target(go),
	  m_variable(val),
	  m_speed(speed),
	  m_duration(duration),
	  m_updateRate(0.016f)
{
	// if a duration is assigned, destroy the thread when timed out
	if (duration > 0) 
	{
		QTimer::singleShot((int)(duration * 1000), this, SLOT(destroy()));
	}

	// add this puppet to the game object
	go->addPuppet(PuppetPtr(this));

	QTimer* timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(update()));
	timer->start(m_updateRate);
}


void Puppet::update()
{
	// only update when the game object exist
	if (m_target)
	{
		// process the required operation
		switch(m_variable)
		{
			case Position:
				m_target->translate(m_speed * m_updateRate);
				break;
			case Rotation:
				m_target->rotate(m_speed * m_updateRate);
				break;
			case Scale:
				m_target->scale(m_speed * m_updateRate);
				break;
		}
	}
	else
		destroy();
}

void Puppet::destroy()
{
	m_target->removePuppet(this);
}

const int Puppet::getVariable() const
{
	return m_variable;
}

const vec3& Puppet::getSpeed() const
{
	return m_speed;
}

const float Puppet::getDuration() const
{
	return m_duration;
}


