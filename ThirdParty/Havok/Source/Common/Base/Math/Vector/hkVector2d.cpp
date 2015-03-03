/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkVector2d.h>

void hkVector2d::setClosestPointOnSegmentToPoint( const hkVector2d& a, const hkVector2d& b, const hkVector2d& p)
{
	hkVector2d edge; edge.setSub(a, b);
	hkVector2d toa; toa.setSub(a,p);
	hkVector2d tob; tob.setSub(b,p);
	hkDouble64 dota = edge.dot(toa);
	hkDouble64 dotb = edge.dot(tob);
	if( dota * dotb < 0 )           
	{
		this->setInterpolate( a, b, dota/(dota-dotb) );
	}
	else if( dota < 0 )
	{
		*this = a;
	}
	else
	{
		*this = b;
	}
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
