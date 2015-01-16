#pragma once
#include "AbstractEmitter.h"

class SphereEmitter : public AbstractEmitter
{
public:
	SphereEmitter();
	~SphereEmitter();

	virtual void emitParticle( Particle& particle );

	float MinimumRadius;
	float MaximumRadius;

	float MinInclination;
	float MaxInclination;

	float MinAzimuth;
	float MaxAzimuth;

	float MinSpeed;
	float MaxSpeed;

	float MinLifetime;
	float MaxLifetime;

	vec3  Origin;

private: 
	void renderSphere( QColor color, float fRadius );
};

