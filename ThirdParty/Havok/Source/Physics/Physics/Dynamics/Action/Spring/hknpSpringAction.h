/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SPRING_ACTION_H
#define HKNP_SPRING_ACTION_H

#include <Physics/Physics/Dynamics/Action/hknpAction.h>


/// A simple spring.
class hknpSpringAction: public hknpBinaryAction
{
	public:
		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_PHYSICS);

		HK_DECLARE_REFLECTION();

		/// Creates a spring.
		hknpSpringAction( hknpBodyId idA, hknpBodyId idB = hknpBodyId(0), hkUlong userData = 0 );

		// Serializing constructor
		hknpSpringAction( class hkFinishLoadedObjectFlag flag ) : hknpBinaryAction(flag) {}

		// hknpAction interface implementation
		virtual void initialize( hknpBodyId idA, hknpBodyId idB = hknpBodyId(0), hkUlong userData = 0 );
		virtual ApplyActionResult applyAction( const hknpSimulationThreadContext& tl, const hknpSolverInfo& stepInfo, hknpCdPairWriter* HK_RESTRICT pairWriter );

		/// Set the connection points to the bodies in world space. Make sure the
		/// body pointers are set before this call.
		void setPositionsInWorldSpace( const hkTransform& bodyATransform, hkVector4Parameter pivotA, const hkTransform& bodyBTransform, hkVector4Parameter pivotB );

		/// Set the connection points to the bodies in body space. Make sure the
		/// body pointers are set before this call.
		void setPositionsInBodySpace( hkVector4Parameter pivotA, hkVector4Parameter pivotB, hkReal restLength );

		///Gets the spring connection position in rigid body A
		inline const hkVector4& getPositionAinA();

		///Gets the spring connection position in rigid body B
		inline const hkVector4& getPositionBinB();

		/// Sets the strength value for the spring.
		inline void setStrength( hkReal str );

		/// Gets the strength value for the spring.
		inline hkReal getStrength();

		/// Sets the damping value for the spring.
		inline void setDamping( hkReal damp );

		/// Gets the damping value for the spring.
		inline hkReal getDamping();

		/// Sets the rest length value for the spring.
		inline void setRestLength( hkReal restLength );

		/// Gets the rest length value for the spring.
		inline hkReal getRestLength();

		/// Sets the compression flag for the spring.
		inline void setOnCompression( hkBool onCompression );

		/// Gets the compression flag for the spring.
		inline hkBool getOnCompression();

		/// Sets the extension flag for the spring.
		inline void setOnExtension( hkBool onExtension );

		/// Gets the extension flag for the spring.
		inline hkBool getOnExtension();

		/// Gets the last force applied by the spring.
		inline const hkVector4& getLastForce();

		virtual void onShiftWorld( hkVector4Parameter offset );

	public:

		static hkReal HK_CALL calcDampingfromDampingRatio( const hkReal& dampingRatio , const hkReal& strength, const hkReal& mass );
		static hkReal HK_CALL calcDampingRatiofromDamping( const hkReal& damping, const hkReal& strength, const hkReal& mass );

	public:

		/// The force applied during the last simulation step.
		hkVector4 m_lastForce;

		/// The point in bodyA where the spring is attached.
		hkVector4 m_positionAinA; //+hk.RangeReal(absmin=-1000.0,absmax=1000.0, softmin=-10.0, softmax=10.0) +hk.Semantics("POSITION") +hk.Ui(group="Pivot", visible=true, label="Pivot")

		/// The point in bodyB where the spring is attach^ed.
		hkVector4 m_positionBinB; //+hk.RangeReal(absmin=-1000.0,absmax=1000.0, softmin=-10.0, softmax=10.0) +hk.Semantics("POSITION") +hk.Ui(visible=true, label="Pivot On Other Body", endGroup=true)

		/// The desired rest length of the spring.
		hkReal	m_restLength; //+default(1.0f) +hk.RangeReal(absmin=0.0,absmax=1000.0, softmin=0.1, softmax=10.0) +hk.Semantics("DISTANCE") +hk.Ui(group="Constants", visible=true, label="Resting Length")

		/// The desired strength of the spring.
		hkReal	m_strength; //+default(1000.0f) +hk.RangeReal(absmin=0.1,absmax=1000.0) +hk.Ui(visible=true)

		/// The damping to be applied to the spring.
		hkReal	m_damping; //+default(0.1f) +hk.RangeReal(absmin=0.1,absmax=1000.0) +hk.Ui(visible=true)

		/// Flag to indicate if the spring acts during compression.
		hkBool	m_onCompression; //+default(true) +hk.Ui(visible=true)

		/// Flag to indicate if the spring acts during extension.
		hkBool	m_onExtension; //+default(true) +hk.Ui(visible=true, endGroup=true)
};

#include <Physics/Physics/Dynamics/Action/Spring/hknpSpringAction.inl>

#endif // HKNP_SPRING_ACTION_H

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
