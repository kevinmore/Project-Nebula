/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>


void HK_CALL hkpRegisterAlternateShapeTypes( hkpCollisionDispatcher* dis )
{
	//
	//	Warning: order is important, later entries override earlier entries
	//
	dis->registerAlternateShapeType( hkcdShapeType::SPHERE,						hkcdShapeType::CONVEX );
	dis->registerAlternateShapeType( hkcdShapeType::TRIANGLE,					hkcdShapeType::CONVEX );
	dis->registerAlternateShapeType( hkcdShapeType::BOX,							hkcdShapeType::CONVEX );
	dis->registerAlternateShapeType( hkcdShapeType::CAPSULE,						hkcdShapeType::CONVEX );
	dis->registerAlternateShapeType( hkcdShapeType::CYLINDER,					hkcdShapeType::CONVEX );
	dis->registerAlternateShapeType( hkcdShapeType::CONVEX_VERTICES,				hkcdShapeType::CONVEX );
	dis->registerAlternateShapeType( hkcdShapeType::CONVEX_TRANSLATE,			hkcdShapeType::CONVEX );
	dis->registerAlternateShapeType( hkcdShapeType::CONVEX_TRANSFORM,			hkcdShapeType::CONVEX );
	dis->registerAlternateShapeType( hkcdShapeType::CONVEX_PIECE,				hkcdShapeType::CONVEX );

	dis->registerAlternateShapeType( hkcdShapeType::TRIANGLE_COLLECTION,			hkcdShapeType::COLLECTION );
	dis->registerAlternateShapeType( hkcdShapeType::LIST,						hkcdShapeType::COLLECTION );
	dis->registerAlternateShapeType( hkcdShapeType::EXTENDED_MESH,				hkcdShapeType::COLLECTION );
	dis->registerAlternateShapeType( hkcdShapeType::COMPRESSED_MESH,				hkcdShapeType::COLLECTION );
	dis->registerAlternateShapeType( hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION, hkcdShapeType::COLLECTION);

	dis->registerAlternateShapeType( hkcdShapeType::MOPP,						hkcdShapeType::BV_TREE );
	dis->registerAlternateShapeType( hkcdShapeType::STATIC_COMPOUND,				hkcdShapeType::BV_TREE );
	dis->registerAlternateShapeType( hkcdShapeType::BV_COMPRESSED_MESH,			hkcdShapeType::BV_TREE );
	dis->registerAlternateShapeType( hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE, hkcdShapeType::BV_TREE);

	dis->registerAlternateShapeType( hkcdShapeType::CONVEX,						hkcdShapeType::SPHERE_REP );

	dis->registerAlternateShapeType( hkcdShapeType::PLANE,						hkcdShapeType::HEIGHT_FIELD );
	dis->registerAlternateShapeType( hkcdShapeType::SAMPLED_HEIGHT_FIELD,		hkcdShapeType::HEIGHT_FIELD );
}

const char* HK_CALL hkGetShapeTypeName( hkpShapeType type )
{
#define X(a) case a: return #a; break
	switch(type)
	{
		X(hkcdShapeType::INVALID);
		X(hkcdShapeType::ALL_SHAPE_TYPES);
		X(hkcdShapeType::CONVEX);
		X(hkcdShapeType::COMPRESSED_MESH);
		X(hkcdShapeType::COLLECTION);
		X(hkcdShapeType::BV_TREE);
		X(hkcdShapeType::SPHERE);
		X(hkcdShapeType::TRIANGLE);
		X(hkcdShapeType::BOX);

		X(hkcdShapeType::CAPSULE);
		X(hkcdShapeType::CYLINDER);
		X(hkcdShapeType::CONVEX_VERTICES);

		X(hkcdShapeType::CONVEX_PIECE);
		
		X(hkcdShapeType::MULTI_SPHERE);
		X(hkcdShapeType::LIST);
		X(hkcdShapeType::CONVEX_LIST);
		X(hkcdShapeType::TRIANGLE_COLLECTION);
		X(hkcdShapeType::MULTI_RAY);
		X(hkcdShapeType::HEIGHT_FIELD);
		X(hkcdShapeType::SAMPLED_HEIGHT_FIELD);
		X(hkcdShapeType::SPHERE_REP);
		X(hkcdShapeType::BV);
		X(hkcdShapeType::PLANE);
		X(hkcdShapeType::MOPP);
		X(hkcdShapeType::TRANSFORM);
		X(hkcdShapeType::CONVEX_TRANSLATE);
		X(hkcdShapeType::CONVEX_TRANSFORM);
		X(hkcdShapeType::EXTENDED_MESH);
		X(hkcdShapeType::STATIC_COMPOUND);
		X(hkcdShapeType::BV_COMPRESSED_MESH);
		X(hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE);
		X(hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION);
		X(hkcdShapeType::MAX_PPU_SHAPE_TYPE);
		X(hkcdShapeType::PHANTOM_CALLBACK);

		X(hkcdShapeType::USER0);
		X(hkcdShapeType::USER1);
		X(hkcdShapeType::USER2);

		default: return HK_NULL;
	}
}

#undef X

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
