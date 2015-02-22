#pragma once;
#include "Particle.h"
#include <QOpenGLFunctions_4_3_Core>

class AbstractEmitter : protected QOpenGLFunctions_4_3_Core
{
public:
	AbstractEmitter() 
	{
		Q_ASSERT(initializeOpenGLFunctions());	
	}
	virtual ~AbstractEmitter() {}
	virtual void emitParticle( Particle& particle ) = 0;
};