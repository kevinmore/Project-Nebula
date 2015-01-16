#include "SphereEmitter.h"

SphereEmitter::SphereEmitter()
	: AbstractEmitter(),
	  MinimumRadius(0),
	  MaximumRadius(1),
	  MinInclination(0),
	  MaxInclination(180),
	  MinAzimuth(0),
	  MaxAzimuth(360),
	  MinSpeed(10),
	  MaxSpeed(20),
	  MinLifetime(3),
	  MaxLifetime(5),
	  Origin(Vector3D::ZERO)
{
}


SphereEmitter::~SphereEmitter()
{
}

void SphereEmitter::emitParticle( Particle& particle )
{
	float inclination = qDegreesToRadians(Random::random(MinInclination, MaxInclination));
	float azimuth = qDegreesToRadians(Random::random(MinAzimuth, MaxAzimuth));

	float radius = Random::random( MinimumRadius, MaximumRadius );
	float speed = Random::random( MinSpeed, MaxSpeed );
	float lifetime = Random::random( MinLifetime, MaxLifetime );

	float sInclination = qSin( inclination );

	float X = sInclination * qCos( azimuth );
	float Y = sInclination * qSin( azimuth );
	float Z = qCos( inclination );

	vec3 vector( X, Y, Z );

	particle.m_position = ( vector * radius ) + Origin;
	particle.m_velocity = vector * speed;

	particle.m_lifeTime = lifetime;
	particle.m_age = 0;
}

void SphereEmitter::renderSphere( QColor color, float fRadius )
{
	float X, Y, Z, inc, azi;

	glColor4f( color.red()/255, color.green()/255, color.blue()/255, color.alpha()/255 );

	glPointSize(2.0f);
	glBegin( GL_POINTS );

	for ( float azimuth = MinAzimuth; azimuth < MaxAzimuth; azimuth += 5.0f )
	{
		for ( float inclination = MinInclination; inclination < MaxInclination; inclination += 5.0f )
		{
			inc = qDegreesToRadians(inclination);
			azi = qDegreesToRadians(azimuth);

			X = fRadius * qSin( inc ) * qCos( azi );
			Y = fRadius * qSin( inc ) * qSin( azi );
			Z = fRadius * qCos( inc );

			glVertex3f(X, Y, Z );
		}

		inc = qDegreesToRadians(MaxInclination);
		azi = qDegreesToRadians(azimuth);

		X = fRadius * qSin( inc ) * qCos( azi );
		Y = fRadius * qSin( inc ) * qSin( azi );
		Z = fRadius * qCos( inc );

		glVertex3f(X, Y, Z );
	}

	inc = qDegreesToRadians(MaxInclination);
	azi = qDegreesToRadians(MaxAzimuth);

	X = MaximumRadius * qSin( inc ) * qCos( azi );
	Y = MaximumRadius * qSin( inc ) * qSin( azi );
	Z = MaximumRadius * qCos( inc );

	glVertex3f(X, Y, Z );

	glEnd();
}

