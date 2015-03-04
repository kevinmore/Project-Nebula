/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Spline/hkxSpline.h>

	// The Max Kochanek-Bartels (KB) splines as per max have  Tension, Continuity  & Bias
	// http://news.povray.org/povray.binaries.tutorials/attachment/%3CXns91B880592482seed7@povray.org%3E/Splines.bas.txt
	// if you want more than just the Hermit of the given tangents ot Beizer below that
	// Would need to export that from Max etc too then


void hkxSpline::evaluateHermite( int section, float t, hkVector4& r)
{
	int nextSection = m_isClosed && (section == m_controlPoints.getSize()-1) ? 0 : section + 1;
	if ( nextSection >= m_controlPoints.getSize() )
	{
		r = m_controlPoints.back().m_position;
		return;
	}

	const ControlPoint C0 = m_controlPoints[section];
	const ControlPoint C1 = m_controlPoints[nextSection];

	hkMatrix4 CP;
	CP.setColumn<0>( C0.m_position );
	CP.setColumn<1>( C1.m_position );
	CP.setColumn<2>( C0.m_tangentOut );
	CP.setColumn<3>( C1.m_tangentOut );
		
	// Hermite basis functions.
	hkVector4 B;
	hkReal tt = t*t;
	hkReal ttt = tt*t;
	B.set( 2.f*ttt - 3.f*tt + 1.f, -2.f*ttt + 3.f*tt, ttt - 2.f*tt + t,  ttt - tt );

	// Do it.
	CP.multiplyVector(B,r);
}


void hkxSpline::evaluateByType( int section, float t, hkVector4& r)
{
	// implicit last section if closed
	int nextSection = m_isClosed && (section == m_controlPoints.getSize()-1) ? 0 : section + 1;
	if ( nextSection >= m_controlPoints.getSize() )
	{
		r = m_controlPoints.back().m_position;
		return;
	}
	
	const ControlPoint C0 = m_controlPoints[section];
	const ControlPoint C1 = m_controlPoints[nextSection];



	//beziers or half beziers
	float omt = 1.0f-t;
	float omt2= omt * omt;
	float omt3 = omt * omt2;

	hkSimdReal tv; tv.setFromFloat( t ); 
	float t2 = t * t;
	float t3 = t * t2;

	if ( (C0.m_outType == LINEAR) && (C1.m_inType == LINEAR))
	{
		// linear line
		r.setInterpolate(C0.m_position, C1.m_position, tv);
		return;
	}

	//pos = position*omt3 + next.position * t3;
	hkSimdReal omt3v; omt3v.setFromFloat(omt3);
	r.setMul( C0.m_position, omt3v);
	
	hkSimdReal t3v; t3v.setFromFloat(t3);
	hkVector4 nextR; nextR.setMul( C1.m_position, t3v);
	r.add(nextR);
  
	//the following allows us to make "half bezier" splines...
	hkSimdReal s; s.setFromFloat(3.0f*t*omt2);
	if ( C0.m_outType != LINEAR )
	{
		//pos += controlVertices[OUT_VECTOR] * (3*t*omt2);  
		hkVector4 tout; tout.setMul( C0.m_tangentOut, s );
		r.add(tout);
	}
	else 
	{
		//pos += position * (3*t*omt2);
		hkVector4 tout; tout.setMul( C0.m_position, s );
		r.add(tout);
	}
	hkSimdReal ss; ss.setFromFloat(3.0f*t2*omt);
	if ( C1.m_inType != LINEAR )
	{
		//pos += next.controlVertices[IN_VECTOR] * (3*t2*omt); 
		hkVector4 tout; tout.setMul( C1.m_tangentIn, ss);
		r.add(tout);
	}
	else 
	{
		//pos += (next.position) * (3*t2*omt);
		hkVector4 tout; tout.setMul( C1.m_position, ss);
		r.add(tout);
	}

//	if(firstDerivative)
//	{
//		*firstDerivative = C0.m_position * (-3 * t2 + 6 * t - 3) +
//			C0.m_tangentOut * (9 * t2 - 12 * t + 3) +
//			C1.m_tangentIn * (-9 * t2 + 6 * t) +
//			C1.m_position * (3 * t2);
//	}

//	if(secondDerivative)
//	{
//		*secondDerivative = C0.m_position * (-6 * t + 6) + 
//			C0.m_tangentOut * (18 * t - 12) + 
//			C1.m_tangentIn * (-18 * t + 6) + 
//			C1.m_position * 6 * t;
//	}
}

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
