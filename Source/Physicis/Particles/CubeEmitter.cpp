#include "CubeEmitter.h"

CubeEmitter::CubeEmitter()
	: MinWidth(-1),
	  MaxWidth(1),
	  MinHeight(-1),
	  MaxHeight(1),
	  MinDepth(-1),
	  MaxDepth(1),
	  MinSpeed(10),
	  MaxSpeed(20),
	  MinLifetime(3),
	  MaxLifetime(5),
	  Origin(Vector3D::ZERO)
{
}


CubeEmitter::~CubeEmitter()
{
}

void CubeEmitter::emitParticle( Particle& particle )
{
	float X = Random::random( MinWidth, MaxWidth );
	float Y = Random::random( MinHeight, MaxHeight );
	float Z = Random::random( MinDepth, MaxDepth );

	float lifetime = Random::random( MinLifetime, MaxLifetime );
	float speed = Random::random( MinSpeed, MaxSpeed );

	vec3 vector( X, Y, Z );

	particle.m_position = vector + Origin;
	particle.m_velocity = vector.normalized() * speed;

	particle.m_lifeTime = lifetime;
	particle.m_age = 0;
}

void CubeEmitter::renderCube( QColor color, float fRadius )
{

}
