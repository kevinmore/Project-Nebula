/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformdUtil.h>

namespace hkSweptTransformUtil
{

#if ! defined(HK_PLATFORM_SPU)
void lerp2( const hkSweptTransformd& sweptTrans, hkSimdDouble64Parameter t, hkTransformd& transformOut )
{
	hkSimdDouble64 r = sweptTrans.getInterpolationValue( t );
	_lerp2(sweptTrans, r, transformOut);
}
void lerp2( const hkSweptTransformd& sweptTrans, hkTime t, hkTransformd& transformOut )
{
	lerp2(sweptTrans, hkSimdDouble64::fromFloat(t), transformOut);
}
#else
void lerp2( const hkSweptTransformd& sweptTrans, hkSimdDouble64Parameter t, hkTransformd& transformOut )
{
	lerp2Ha( sweptTrans, t.getReal(), hkDouble64(0), transformOut );
}
void lerp2( const hkSweptTransformd& sweptTrans, hkTime t, hkTransformd& transformOut )
{
	lerp2Ha( sweptTrans, t, hkDouble64(0), transformOut );
}
#endif

void lerp2Ha( const hkSweptTransformd& sweptTrans, hkSimdDouble64Parameter t, hkSimdDouble64Parameter tAddOn, hkTransformd& transformOut )
{
	const hkSimdDouble64 r = sweptTrans.getInterpolationValueHiAccuracy( t, tAddOn );
	_lerp2(sweptTrans, r, transformOut);
}

void lerp2Ha( const hkSweptTransformd& sweptTrans, hkTime t, hkDouble64 tAddOn, hkTransformd& transformOut )
{
	lerp2Ha(sweptTrans, hkSimdDouble64::fromFloat(t), hkSimdDouble64::fromFloat(tAddOn), transformOut);
}

#if !defined (HK_PLATFORM_SPU)
void lerp2Rel( const hkSweptTransformd& sweptTrans, hkSimdDouble64Parameter r, hkTransformd& transformOut )
{
	_lerp2(sweptTrans, r, transformOut);
}

void lerp2Rel( const hkSweptTransformd& sweptTrans, hkDouble64 r, hkTransformd& transformOut )
{
	_lerp2(sweptTrans, hkSimdDouble64::fromFloat(r), transformOut);
}

#	if defined(HK_REAL_IS_DOUBLE)
	// only updates t1 and 'hkTransformd' of the hkSweptTransformd
void backStepMotionState( hkSimdDouble64Parameter time, hkMotionState& motionState )
{
	hkSweptTransformd& st = motionState.getSweptTransform();
	hkSimdDouble64 t; t.setMax( st.getInterpolationValue( time ), hkSimdReal_Eps );

	_lerp2( st, t, st.m_rotation1 );
	const hkSimdDouble64 newInvDeltaTime = st.getInvDeltaTimeSr() / t;

	st.m_centerOfMass1.setInterpolate( st.m_centerOfMass0, st.m_centerOfMass1, t );
	st.m_centerOfMass1.setComponent<3>(newInvDeltaTime);
	motionState.m_deltaAngle.mul( t );

	calcTransAtT1( st, motionState.getTransform());
}

void backStepMotionState( hkTime time, hkMotionState& motionState )
{
	backStepMotionState(hkSimdDouble64::fromFloat(time),motionState);
}

	// resets both t0 and t1 transforms of the hkSweptTransformd to the same value and 
	// sets invDeltaTime to zero
void freezeMotionState( hkSimdDouble64Parameter time, hkMotionState& motionState )
{
	hkSweptTransformd& st = motionState.getSweptTransform();
	HK_ASSERT2(0xf0ff0082, st.getInvDeltaTime() == hkDouble64(0) || (( time  * st.getInvDeltaTimeSr() ).getReal() <= ( st.getBaseTime() * st.getInvDeltaTime() ) + hkDouble64(2) ) , "Inconsistent time in motion state.");

		// we actually freeze the object at the earliest moment (defined by hkSweptTransformd.m_startTime) after 'time'
	hkSimdDouble64 maxTime; maxTime.setMax(time, st.getBaseTimeSr());
	const hkSimdDouble64 t = st.getInterpolationValue( maxTime );

	_lerp2( st, t, st.m_rotation1 );
	st.m_rotation0 = st.m_rotation1;

	st.m_centerOfMass1.setInterpolate( st.m_centerOfMass0, st.m_centerOfMass1, t );
	st.m_centerOfMass0 = st.m_centerOfMass1;

	// set time information
	st.m_centerOfMass0.setComponent<3>(maxTime);
	st.m_centerOfMass1.zeroComponent<3>();

	calcTransAtT1( st, motionState.getTransform());
}

void freezeMotionState( hkTime time, hkMotionState& motionState )
{
	freezeMotionState(hkSimdDouble64::fromFloat(time),motionState);
}

void setTimeInformation( hkTime startTime, hkDouble64 invDeltaTime, hkMotionState& motionState)
{
	motionState.getSweptTransform().m_centerOfMass0(3) = startTime;
	motionState.getSweptTransform().m_centerOfMass1(3) = invDeltaTime;
}

void setTimeInformation( hkSimdDouble64Parameter startTime, hkSimdDouble64Parameter invDeltaTime, hkMotionState& motionState)
{
	motionState.getSweptTransform().m_centerOfMass0.setComponent<3>(startTime);
	motionState.getSweptTransform().m_centerOfMass1.setComponent<3>(invDeltaTime);
}

void warpTo( hkVector4dParameter position, hkQuaterniondParameter rotation, hkMotionState& ms )
{
	hkSweptTransformd& sweptTransform = ms.getSweptTransform();
	ms.m_deltaAngle.setZero();

	sweptTransform.m_rotation0 = rotation;
	sweptTransform.m_rotation1 = rotation;

	ms.getTransform().setRotation( rotation );
	ms.getTransform().setTranslation( position );
	
	hkVector4d centerShift;
	centerShift._setRotatedDir( ms.getTransform().getRotation(), sweptTransform.m_centerOfMassLocal );

	const hkSimdDouble64 baseTime = sweptTransform.getBaseTimeSr();

	sweptTransform.m_centerOfMass0.setAdd( position, centerShift );
	sweptTransform.m_centerOfMass1 = sweptTransform.m_centerOfMass0;

	sweptTransform.m_centerOfMass0.setComponent<3>(baseTime);	
	sweptTransform.m_centerOfMass1.zeroComponent<3>();	// invDeltaTime
}

void warpTo( const hkTransformd& transform, hkMotionState& ms )
{
	hkSweptTransformd& sweptTransform = ms.getSweptTransform();
	ms.m_deltaAngle.setZero();

	hkQuaterniond rotation; rotation.set( transform.getRotation() );
	ms.getTransform() = transform;

	sweptTransform.m_rotation0 = rotation;
	sweptTransform.m_rotation1 = rotation;
	
	hkVector4d centerShift;
	centerShift._setRotatedDir( transform.getRotation(), sweptTransform.m_centerOfMassLocal );

	const hkSimdDouble64 baseTime = sweptTransform.getBaseTimeSr();

	sweptTransform.m_centerOfMass0.setAdd( transform.getTranslation(), centerShift );
	sweptTransform.m_centerOfMass1 = sweptTransform.m_centerOfMass0;

	sweptTransform.m_centerOfMass0.setComponent<3>(baseTime);	
	sweptTransform.m_centerOfMass1.zeroComponent<3>();	// invDeltaTime
}

void warpToPosition( hkVector4dParameter position, hkMotionState& ms )
{
	const hkRotationd& currentRotation = ms.getTransform().getRotation();
	hkSweptTransformd& sweptTransform = ms.getSweptTransform();

	ms.m_deltaAngle.setZero();
	ms.getTransform().setTranslation( position );

	hkVector4d centerShift;
	centerShift._setRotatedDir( currentRotation, sweptTransform.m_centerOfMassLocal );

	const hkSimdDouble64 baseTime = sweptTransform.getBaseTimeSr();

	sweptTransform.m_centerOfMass0.setAdd( position, centerShift );
	sweptTransform.m_centerOfMass1 = sweptTransform.m_centerOfMass0;

	sweptTransform.m_rotation0 = sweptTransform.m_rotation1;

	sweptTransform.m_centerOfMass0.setComponent<3>(baseTime);	
	sweptTransform.m_centerOfMass1.zeroComponent<3>(); // invDeltaTime
}

void warpToRotation( hkQuaterniondParameter rotation, hkMotionState& ms )
{
	warpTo( ms.getTransform().getTranslation(), rotation, ms );
}

void keyframeMotionState( const hkStepInfo& stepInfo, hkVector4dParameter pos1, hkQuaterniondParameter rot1, hkMotionState& ms )
{
	hkSweptTransformd& sweptTransform = ms.getSweptTransform();

	sweptTransform.m_centerOfMass0 = sweptTransform.m_centerOfMass1;
	sweptTransform.m_rotation0 = sweptTransform.m_rotation1;

	sweptTransform.m_centerOfMass1 = pos1;
	sweptTransform.m_rotation1 = rot1;

	hkQuaterniond diff; diff.setMulInverse( rot1, sweptTransform.m_rotation0 );

	hkSimdDouble64 angle; angle.setFromFloat(diff.getAngle());
	hkVector4d axis;
	if ( diff.hasValidAxis() )
	{
		diff.getAxis(axis);
	}
	else
	{
		axis.setZero();
	}

	ms.m_deltaAngle.setMul( angle, axis );
	ms.m_deltaAngle.setComponent<3>(angle);
	
	//
	//	Use the angle to calculate redundant information
	//
	/*
	{
		const hkDouble64 angle2 = angle * angle;
		const hkDouble64 angle3 = angle * angle2;
		const hkDouble64 sa = 0.044203f;
		const hkDouble64 sb = 0.002343f;

			// this is:
			// 2.0f * sin( 0.5f * angle ) / angle
			// and can be used as a factor to m_deltaAngle
			// to get m_deltaAngleLower or the maximum projected distance any
			// point on the unit sphere of the object can travel 
		const hkDouble64 rel2SinHalfAngle = 1.0f - sa * angle2 + sb * angle3;
		const hkDouble64 collisionToleranceEps = 0.01f * 0.01f;
		ms.m_maxAngularError = collisionToleranceEps + ms.m_objectRadius * rel2SinHalfAngle * angle;
	}
	*/

	sweptTransform.m_centerOfMass0(3) = stepInfo.m_startTime.val();	
	sweptTransform.m_centerOfMass1(3) = stepInfo.m_invDeltaTime.val();

	calcTransAtT1( sweptTransform, ms.getTransform() );
}

void setCentreOfRotationLocal( hkVector4dParameter newCenterOfRotation, hkMotionState& motionState)
{
	hkSweptTransformd& st = motionState.getSweptTransform();

	hkVector4d offset; offset.setSub(newCenterOfRotation, st.m_centerOfMassLocal);
	st.m_centerOfMassLocal = newCenterOfRotation;
	
	hkVector4d offsetWs; offsetWs._setRotatedDir(motionState.getTransform().getRotation(), offset);

	const hkSimdDouble64 t0 = st.m_centerOfMass0.getComponent<3>();
	const hkSimdDouble64 t1 = st.m_centerOfMass1.getComponent<3>();

	st.m_centerOfMass0.addXYZ(offsetWs);
	st.m_centerOfMass1.addXYZ(offsetWs);

	st.m_centerOfMass0.setComponent<3>(t0);
	st.m_centerOfMass1.setComponent<3>(t1);
}
#endif // spu

#endif // defined(HK_REAL_IS_DOUBLE)

// Has to be outside of inline function as gcc won't inline functions with statics in them.
const hkQuadDouble64 _stepMotionStateMaxVeld = HK_QUADDOUBLE_CONSTANT(1e6f,1e6f,1e6f,1e6f);

} // namespace hkSweptTransformUtil

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
