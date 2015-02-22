#pragma once
#include "AbstractEmitter.h"

class CubeEmitter : public AbstractEmitter
{
public:
	CubeEmitter();
	~CubeEmitter();

	virtual void emitParticle( Particle& particle );

	float MinWidth;
	float MaxWidth;

	float MinHeight;
	float MaxHeight;

	float MinDepth;
	float MaxDepth;

	float MinSpeed;
	float MaxSpeed;

	float MinLifetime;
	float MaxLifetime;

	vec3  Origin;

private: 
	void renderCube( QColor color, float fRadius );
};

