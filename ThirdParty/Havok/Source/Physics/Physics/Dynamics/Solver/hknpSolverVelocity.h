/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SOLVER_VELOCITY_H
#define HKNP_SOLVER_VELOCITY_H

#include <Common/Base/Math/Vector/hkVector4.h>
#include <Common/Base/Math/Vector/Mx/hkMxVector.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionCinfo.h>


/// Holds solver velocity data using a minimal amount of memory.
/// Provides functions to set and get from the more performant data structures used during solving (like hkMxVector).
HK_CLASSALIGN(class,HK_REAL_ALIGNMENT) hknpSolverVelocity
{
	public:

		/// Performant initialization of all the data
		HK_FORCE_INLINE void init();

		/// Gets the linear velocity
		HK_FORCE_INLINE void getLinearVelocity(hkVector4& velOut) const;
		/// Sets the linear velocity
		HK_FORCE_INLINE void setLinearVelocity(hkVector4Parameter v);

		/// Gets the linear velocities from an array of hknpSolverVelocity to a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getLinearVelocities(const hknpSolverVelocity* velocities, MXVECTOR& linVelsOut);
		/// Sets the linear velocities of an array of hknpSolverVelocity from a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setLinearVelocities(const MXVECTOR& linVels, hknpSolverVelocity* velsOut);
		/// Gets the linear velocities from an array of hknpSolverVelocity pointers to a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getLinearVelocities(const hknpSolverVelocity* velocities[], MXVECTOR& linVelsOut);
		/// Sets the linear velocities of an array of hknpSolverVelocity pointers from a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setLinearVelocities(const MXVECTOR& linVels, hknpSolverVelocity** velsOut);

		/// Gets the angular velocity
		HK_FORCE_INLINE void getAngularVelocity(hkVector4& velOut) const;
		/// Sets the angular velocity
		HK_FORCE_INLINE void setAngularVelocity(hkVector4Parameter v);

		/// Gets the angular velocities from an array of hknpSolverVelocity to a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getAngularVelocities(const hknpSolverVelocity* velocities, MXVECTOR& angVelsOut);
		/// Sets the angular velocities of an array of hknpSolverVelocity from a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setAngularVelocities(const MXVECTOR& angVels, hknpSolverVelocity* velocities);
		/// Gets the angular velocities from an array of hknpSolverVelocity pointers to a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getAngularVelocities(const hknpSolverVelocity* velocities[], MXVECTOR& angVelsOut);
		/// Sets the angular velocities of an array of hknpSolverVelocity pointers from a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setAngularVelocities(const MXVECTOR& angVels, hknpSolverVelocity** velsOut);


		/// Sets both velocities (faster than calling the 2 individual functions)
		HK_FORCE_INLINE void setVelocity(hkVector4Parameter linVel, hkVector4Parameter angVel);
		/// Gets both velocities from an array of hknpSolverVelocity to a hkMxVector (faster than calling the 2 individual functions)
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getVelocities(const hknpSolverVelocity* velocities, MXVECTOR& linVelsOut, MXVECTOR& angVelsOut);
		/// Sets both velocities of an array of hknpSolverVelocity from a hkMxVector (faster than calling the 2 individual functions)
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setVelocities(const MXVECTOR& linVels, const MXVECTOR& angVels, hknpSolverVelocity* velsOut);
		/// Gets both velocities from an array of hknpSolverVelocity pointers to a hkMxVector (faster than calling the 2 individual functions)
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getVelocities(const hknpSolverVelocity* velocities[], MXVECTOR& linVelsOut, MXVECTOR& angVelsOut);
		/// Sets both velocities of an array of hknpSolverVelocity pointers from a hkMxVector (faster than calling the 2 individual functions)
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setVelocities(const MXVECTOR& linVels, const MXVECTOR& angVels, hknpSolverVelocity** velsOut);

		/// Gets all the data from an array of hknpSolverVelocity pointers to a hkMxVector (faster than calling the 3 individual functions)
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getAll(const hknpSolverVelocity* velocities[], MXVECTOR& linVelsOut, MXVECTOR& angVelsOut, MXVECTOR& invIntertiasOut);

		// Sets the inverse moment of inertia and inverse mass inertia from an array of 4 halfs
		HK_FORCE_INLINE	void setInvInertias(const hkHalf* h);
		// Gets the inverse moment of inertia (components 0,1,2) and inverse mass inertia (component 3)
		HK_FORCE_INLINE	void getInvInertias(hkVector4& intertias) const;

		// Gets the inverse inertias from an array of hknpSolverVelocity pointers to a hkMxVector
		template<typename MXVECTOR>
		static HK_FORCE_INLINE void	getInvInertias(const hknpSolverVelocity* velocities[], MXVECTOR& invIntertiasOut);
		// Sets the inverse inertias of an array of hknpSolverVelocity pointers from a hkMxVector
		template<typename MXVECTOR>
		static HK_FORCE_INLINE void	setInvInertias(const MXVECTOR& invIntertias, hknpSolverVelocity** velsOut);

	private:

		// Private functions for getting pointers to data with the correct data types.

		HK_FORCE_INLINE hkVector4* linearVelocity();
		HK_FORCE_INLINE const hkVector4* linearVelocity() const;
		HK_FORCE_INLINE hkReal* angularVelocity();
		HK_FORCE_INLINE const hkReal* angularVelocity() const;

		HK_FORCE_INLINE hkVector4* vector0();
		HK_FORCE_INLINE hkVector4* vector1();
		HK_FORCE_INLINE const hkVector4* vector0() const;
		HK_FORCE_INLINE const hkVector4* vector1() const;

	private:

		// The data, packed for minimal memory usage.

		hkReal	m_linearVelocity[3];	// This has to have an alignment appropriate for hkVector4 SIMD.
		hkReal	m_angularVelocity[3];
		hkHalf	m_invInertiaTensor[3];
		hkHalf	m_invMassIntertia;

#if defined(HK_REAL_IS_DOUBLE)
		hkUint8	m_padding1[8];			// We pad up to the size of 2 hkVector4 to be able to perform a fast setZero().
#endif

		typedef hkReal AngularVelocityDataType;
		typedef hkHalf InvInertiasDataType;
};


/// Holds solver velocity sum data using a minimal amount of memory.
/// Provides functions to set and get from the more performant data structures used during solving (like hkMxVector)
HK_CLASSALIGN(class,HK_REAL_ALIGNMENT) hknpSolverSumVelocity
{
	public:

		/// Performant initialization of all the data
		HK_FORCE_INLINE void init();

		/// Gets the linear velocity
		HK_FORCE_INLINE void getLinearVelocity(hkVector4& velOut) const;
		/// Sets the linear velocity
		HK_FORCE_INLINE void setLinearVelocity(hkVector4Parameter v);

		/// Gets the linear velocities from an array of hknpSolverSumVelocity to a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getLinearVelocities(const hknpSolverSumVelocity* velocities, MXVECTOR& linVelsOut);
		/// Sets the linear velocities of an array of hknpSolverSumVelocity from a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setLinearVelocities(const MXVECTOR& linVels, hknpSolverSumVelocity* velsOut);
		/// Gets the linear velocities from an array of hknpSolverVelocity pointers to a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getLinearVelocities(const hknpSolverSumVelocity* velocities[], MXVECTOR& linVelsOut);
		/// Sets the linear velocities of an array of hknpSolverVelocity pointers from a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setLinearVelocities(const MXVECTOR& linVels, hknpSolverSumVelocity** velsOut);

		/// Gets the angular velocity
		HK_FORCE_INLINE void getAngularVelocity(hkVector4& velOut) const;
		/// Sets the angular velocity
		HK_FORCE_INLINE void setAngularVelocity(hkVector4Parameter v);

		/// Gets the angular velocities from an array of hknpSolverVelocity to a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getAngularVelocities(const hknpSolverSumVelocity* velocities, MXVECTOR& angVelsOut);
		/// Sets the angular velocities of an array of hknpSolverVelocity from a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setAngularVelocities(const MXVECTOR& angVels, hknpSolverSumVelocity* velocities);
		/// Gets the angular velocities from an array of hknpSolverVelocity pointers to a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getAngularVelocities(const hknpSolverSumVelocity* velocities[], MXVECTOR& angVelsOut);
		/// Sets the angular velocities of an array of hknpSolverVelocity pointers from a hkMxVector
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setAngularVelocities(const MXVECTOR& angVels, hknpSolverSumVelocity** velsOut);

		/// Sets both velocities (faster than calling the 2 individual functions)
		HK_FORCE_INLINE void setVelocity(hkVector4Parameter linVel, hkVector4Parameter angVel);
		/// Gets both velocities from an array of hknpSolverVelocity to a hkMxVector (faster than calling the 2 individual functions)
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getVelocities(const hknpSolverSumVelocity* velocities, MXVECTOR& linVelsOut, MXVECTOR& angVelsOut);
		/// Sets both velocities of an array of hknpSolverVelocity from a hkMxVector (faster than calling the 2 individual functions)
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setVelocities(const MXVECTOR& linVels, const MXVECTOR& angVels, hknpSolverSumVelocity* velsOut);
		/// Gets both velocities from an array of hknpSolverVelocity pointers to a hkMxVector (faster than calling the 2 individual functions)
		template<typename MXVECTOR> static HK_FORCE_INLINE void	getVelocities(const hknpSolverSumVelocity* velocities[], MXVECTOR& linVelsOut, MXVECTOR& angVelsOut);
		/// Sets both velocities of an array of hknpSolverVelocity pointers from a hkMxVector (faster than calling the 2 individual functions)
		template<typename MXVECTOR> static HK_FORCE_INLINE void	setVelocities(const MXVECTOR& linVels, const MXVECTOR& angVels, hknpSolverSumVelocity** velsOut);

		/// Gets the maximum angular speed that should prevent tunneling
		HK_FORCE_INLINE const hkHalf& getAngularSpeedLimit() const;
		/// Sets the maximum angular speed that should prevent tunneling
		HK_FORCE_INLINE void setAngularSpeedLimit(hkHalf value);
		/// Sets the maximum angular speed that should prevent tunneling
		HK_FORCE_INLINE void setAngularSpeedLimit(hkSimdRealParameter value);

		/// Gets the motion Id that this structure's data was created from
		HK_FORCE_INLINE const hknpMotionId& getOriginalMotionId() const;
		/// Sets the motion Id that this structure's data was created from
		HK_FORCE_INLINE void setOriginalMotionId(hknpMotionId value);

		/// Gets the motion properties ID
		HK_FORCE_INLINE const hknpMotionPropertiesId& getMotionPropertiesId() const;
		/// Sets the motion properties ID
		HK_FORCE_INLINE void setMotionPropertiesId(hknpMotionPropertiesId value);

	public:

		// Private functions for getting pointers to data with the correct data types.

		HK_FORCE_INLINE hkVector4* linearVelocity();
		HK_FORCE_INLINE const hkVector4* linearVelocity() const;
		HK_FORCE_INLINE hkReal* angularVelocity();
		HK_FORCE_INLINE const hkReal* angularVelocity() const;

		HK_FORCE_INLINE hkVector4* vector0();
		HK_FORCE_INLINE hkVector4* vector1();
		HK_FORCE_INLINE const hkVector4* vector0() const;
		HK_FORCE_INLINE const hkVector4* vector1() const;

	private:

		// The data, packed for minimal memory usage.

		hkReal m_linearVelocity[3];	// This has to have an alignment appropriate for hkVector4 SIMD.
		hkReal m_angularVelocity[3];

		hknpMotionId m_originalMotionId;
		hknpMotionPropertiesId m_motionPropertiesId;
		hkHalf m_angularSpeedLimit;

#if defined(HK_REAL_IS_DOUBLE)
		hkUint8 m_padding1[8];			// We pad up to the size of 2 hkVector4 to be able to perform a fast setZero().
#endif

		typedef hkReal AngularVelocityDataType;
};


#include <Physics/Physics/Dynamics/Solver/hknpSolverVelocity.inl>


#endif // HKNP_SOLVER_VELOCITY_H

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
