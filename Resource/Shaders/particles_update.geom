#version 430

layout(points) in;
layout(points) out;
layout(max_vertices = 40) out;

// All that we get from vertex shader

in vec3 vPositionPass[];
in vec3 vVelocityPass[];
in vec3 vColorPass[];
in float fLifeTimePass[];
in float fSizePass[];
in int iTypePass[];

// All that we send further

out vec3 vPositionOut;
out vec3 vVelocityOut;
out vec3 vColorOut;
out float fLifeTimeOut;
out float fSizeOut;
out int iTypeOut;

uniform float fParticleMass; // Mass of the particle
uniform float fGravityFactor; // Factor which determines how much gravity would influence the particle
uniform vec3 vGenPosition; // Position where new particles are spawned
uniform vec3 vForce; // Gravity vector for particles - updates velocity of particles 
uniform int iCollisionEnabled; // 1 means enable collision detection, 0 means disable
uniform float fRestitution; // Determines how much veclocity is preserved when collided
uniform vec3 vPlaneNormal; // The normal vector of the plane for collision detection
uniform vec3 vPlanePoint; // An arbitury point on the plane
uniform vec3 vGenVelocityMin; // Velocity of new particle - from min to (min+range)
uniform vec3 vGenVelocityRange;

uniform vec3 vGenColor;
uniform float fGenSize; 

uniform float fGenLifeMin, fGenLifeRange; // Life of new particle - from min to (min+range)
uniform float fTimePassed; // Time passed since last frame

uniform vec3 vRandomSeed; // Seed number for our random number function
vec3 vLocalSeed;

uniform int iNumToGenerate; // How many particles will be generated next time, if greater than zero, particles are generated

// This function returns random number from zero to one
float randZeroOne()
{
    uint n = floatBitsToUint(vLocalSeed.y * 214013.0 + vLocalSeed.x * 2531011.0 + vLocalSeed.z * 141251.0);
    n = n * (n * n * 15731u + 789221u);
    n = (n >> 9u) | 0x3F800000u;
 
    float fRes =  2.0 - uintBitsToFloat(n);
    vLocalSeed = vec3(vLocalSeed.x + 147158.0 * fRes, vLocalSeed.y*fRes  + 415161.0 * fRes, vLocalSeed.z + 324154.0*fRes);
    return fRes;
}

void main()
{
  vLocalSeed = vRandomSeed;
  
  // gl_Position doesn't matter now, as rendering is discarded, so I don't set it at all

  vPositionOut = vPositionPass[0];
  vVelocityOut = vVelocityPass[0];

  // a normal particle
  if(iTypePass[0] != 0)
  {
	vec3 acc = fGravityFactor * vec3(0, -9.8, 0);
	acc += vForce / (fParticleMass + 0.00001); // calculate the accelaration(in case the mass is 0)
	vVelocityOut += acc * fTimePassed; // update the velocity
    vPositionOut += vVelocityOut * fTimePassed; // update the position

	// collision handling (assume it's the x-z plane)
	if(iCollisionEnabled == 1)
	{
		// velocity projection on the normal direction
		float fVelocityNormalProjection = dot(vPlaneNormal, vVelocityOut);
		float fPositionNormalProjection = dot(vPositionOut - vPlanePoint, vPlaneNormal);
		if( fPositionNormalProjection < 0.01 && fVelocityNormalProjection < 0.01)
		{
			//mv1^2 = k*mv0^2
			vVelocityOut = sqrt(fRestitution) * (vVelocityOut - 2 * fVelocityNormalProjection * vPlaneNormal);
		}
	}
  }

  vColorOut = vColorPass[0];
  fLifeTimeOut = fLifeTimePass[0]-fTimePassed;
  fSizeOut = fSizePass[0];
  iTypeOut = iTypePass[0];
    
  // generator
  if(iTypeOut == 0)
  {
    EmitVertex();
    EndPrimitive();
    
    for(int i = 0; i < iNumToGenerate; i++)
    {
      vPositionOut = vGenPosition;
      vVelocityOut = vGenVelocityMin+vec3(vGenVelocityRange.x*randZeroOne(), vGenVelocityRange.y*randZeroOne(), vGenVelocityRange.z*randZeroOne());
      vColorOut = vGenColor;
      fLifeTimeOut = fGenLifeMin+fGenLifeRange*randZeroOne();
      fSizeOut = fGenSize;
      iTypeOut = 1;
      EmitVertex();
      EndPrimitive();
    }
  }
  // normal particle
  else if(fLifeTimeOut > 0.0)
  {
      EmitVertex();
      EndPrimitive(); 
  }
}