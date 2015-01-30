#include "Puppet.h"
#include <QTimer>

Puppet::Puppet( GameObjectPtr go, Variable val, float amount, float duration /*= 0.0f*/ )
	: m_target(go),
	  m_variable(val),
	  m_amount(amount),
	  m_duration(duration)
{
	// if a duration is assigned, destroy the thread when timed out
	if (duration > 0) 
	{
		QTimer::singleShot((int)(duration * 1000), this, SLOT(deleteLater()));
	}

	QTimer* timer = new QTimer(this);
	connect(timer, SIGNAL(timeout()), this, SLOT(update()));
	timer->start(0.016); // f = 1 / 16.10e-3 = 60Hz
}


void Puppet::update()
{
	// only update when the game object exist
	if (m_target)
	{
		m_target->rotateY(5 * 0.016f);
	}
	else
		deleteLater();
}


