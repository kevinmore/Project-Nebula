#pragma once

#include <Animation/Rig/Skeleton.h>

class CCDIKSolver
{
public:

	// A constraint to be solved by the hkIkSolver. A constraint specified a chain of bones
	// and a desired world transformation for the end bone (end effector)
	struct IkConstraint
	{
		// The start bone in this chain
		Bone* m_startBone;

		// The end bone in this chain
		Bone* m_endBone;

		// The target position for the end bone, in model space
		vec3 m_targetMS;

	};

	// Constructor, it takes the number of iterations used by the solver.
	CCDIKSolver( int iterations = 128 );

	~CCDIKSolver();


	// The main method in the class, it performs the IK on the given skeleton.
	bool solve ( const QVector<IkConstraint>& constraints, Skeleton* skeleton );

	bool solveOneConstraint(const IkConstraint& constraint, Skeleton* skeleton);

	// Sets the number of iterations
	void setIterations( int number ) { if (number > 0) m_iterations = number; }

	// Gets the number of iterations
	uint getIterations() const { return m_iterations; }
	
	void getBoneTransforms( Skeleton* skeleton, Bone* baseBone, QVector<mat4>& Transforms );

protected:

	// The number of iterations of the IK solver
	int m_iterations;

	// The last distance between the effector and the target
	float m_lastDistance;
};

