/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLISION_RESULT_ACCESSORS_H
#define HKNP_COLLISION_RESULT_ACCESSORS_H


/// A collection of accessor methods to a RAY CAST query's result data.
struct hknpRayCastQueryResult : public hknpCollisionResult
{
	public:

		/// Returns the world space position where the ray has hit the shape.
		HK_FORCE_INLINE const hkVector4& getPosition() const;

		/// Returns the world space surface normal where the ray has hit the shape.
		HK_FORCE_INLINE const hkVector4& getSurfaceNormal() const;

		/// Returns the hit position along the ray as a fraction of the original ray's length.
		HK_FORCE_INLINE hkReal getFraction( const hknpRayCastQuery& query ) const;

		/// Returns the absolute distance between the ray's origin and the hit position.
		HK_FORCE_INLINE hkReal getDistance( const hknpRayCastQuery& query ) const;

		/// Returns the raw fraction as returned by the ray cast.
		/// Note that depending on the ray cast setup this value can be:
		/// * in the range of [0,1] if the ray's length was baked into the ray's direction or
		/// * the actual distance (as returned by getHitDistance()) if the ray's direction was normalized.
		HK_FORCE_INLINE hkReal getFractionRaw() const;

		/// Returns the hit body ID.
		/// Can be HK_NULL for calls to hknpShapeQueryInterface::castRay() if no body information was provided.
		HK_FORCE_INLINE hknpBodyId getBodyId() const;

		/// Returns the shape key of the hit leaf shape.
		HK_FORCE_INLINE hknpShapeKey getShapeKey() const;

		/// Returns the material id of the hit leaf shape.
		HK_FORCE_INLINE hknpMaterialId getShapeMaterialId() const;

		/// Returns the collision filter info of the hit leaf shape.
		HK_FORCE_INLINE hkUint32 getShapeCollisionFilterInfo() const;

		/// Returns the user data of the hit leaf shape.
		HK_FORCE_INLINE hkUint64 getShapeUserData() const;

		/// Returns TRUE if the ray has hit the shape on its inside.
		/// Inner hits reporting has to be explicitly enabled by setting the hknpRayCastQuery::m_flags
		/// to ENABLE_INSIDE_HITS.
		/// An inner hit can occur only if the ray started inside a shape and only for that first shape. Any subsequent
		/// reported hits will be outer hits.
		HK_FORCE_INLINE hkBool32 isInnerHit() const;
};


/// A collection of accessor methods to a SHAPE CAST query's result data.
struct hknpShapeCastQueryResult : public hknpCollisionResult
{
	public:

		/// Returns the world space contact position where the query shape collided with the hit shape.
		HK_FORCE_INLINE const hkVector4& getContactPosition() const;

		/// Returns the world space contact normal, pointing from the hit shape to the query shape.
		HK_FORCE_INLINE const hkVector4& getHitShapeContactNormal() const;

		/// Returns the world space contact normal, pointing from the query shape to the hit shape.
		
		HK_FORCE_INLINE void getQueryShapeContactNormal( hkVector4* normal ) const;

		/// Returns the query shape's position (upon contact) along the cast direction as a fraction of the original
		/// casting length.
		HK_FORCE_INLINE hkReal getFraction( const hknpShapeCastQuery& query ) const;

		/// Returns the absolute distance between the cast's origin and the query shape's position upon contact.
		HK_FORCE_INLINE hkReal getDistance( const hknpShapeCastQuery& query ) const;

		/// Returns the raw fraction as returned by the shape cast.
		/// Note that depending on the shape cast setup this value can be:
		/// * in the range of [0,1] if the casting length was baked into the casting direction or
		/// * the actual distance (as returned by getHitDistance()) if the casting direction was normalized.
		HK_FORCE_INLINE hkReal getFractionRaw() const;

		/// Returns the query shape's associated body ID.
		/// Can be HK_NULL for calls to hknpWorld::castShape() and hknpShapeQueryInterface::castShape() if
		/// no body information was provided.
		HK_FORCE_INLINE hknpBodyId getQueryBodyId() const;

		/// Returns the shape key of the query shape's colliding leaf shape.
		HK_FORCE_INLINE hknpShapeKey getQueryShapeKey() const;

		/// Returns the material id of the query shape's colliding leaf shape.
		HK_FORCE_INLINE hknpMaterialId getQueryShapeMaterialId() const;

		/// Returns the collision filter info of the query shape's colliding leaf shape.
		HK_FORCE_INLINE hkUint32 getQueryShapeCollisionFilterInfo() const;

		/// Returns the user data of the query shape's colliding leaf shape.
		HK_FORCE_INLINE hkUint64 getQueryShapeUserData() const;

		/// Returns the hit shape's associated body ID.
		/// Can be HK_NULL for calls to hknpShapeQueryInterface::castShape() if no body information was provided.
		HK_FORCE_INLINE hknpBodyId getHitBodyId() const;

		/// Returns the shape key of the hit shape's colliding leaf shape.
		HK_FORCE_INLINE hknpShapeKey getHitShapeKey() const;

		/// Returns the material id of the hit shape's colliding leaf shape.
		HK_FORCE_INLINE hknpMaterialId getHitShapeMaterialId() const;

		/// Returns the collision filter info of the hit shape's colliding leaf shape.
		HK_FORCE_INLINE hkUint32 getHitShapeCollisionFilterInfo() const;

		/// Returns the user data of the hit shape's colliding leaf shape.
		HK_FORCE_INLINE hkUint64 getHitShapeUserData() const;
};


/// A collection of accessor methods to a CLOSEST POINTS query's result data.
struct hknpClosestPointsQueryResult : public hknpCollisionResult
{
	public:

		/// Returns the world space position of the point on the hit shape that is closest to the query shape.
		HK_FORCE_INLINE const hkVector4& getClosestPointOnHitShape() const;

		/// Returns the world space position of the point on the query shape that is closest to the hit shape.
		HK_FORCE_INLINE void getClosestPointOnQueryShape( hkVector4* point ) const;

		/// Returns the normalized separating vector in world space, pointing from hit shape to query shape.
		HK_FORCE_INLINE const hkVector4& getSeparatingDirection() const;

		/// Returns the shortest distance between the query shape and the hit shape.
		HK_FORCE_INLINE hkReal getDistance() const;

		/// Returns the query shape's associated body ID.
		/// Can be HK_NULL for calls to hknpWorld::getClosestPoints() and hknpShapeQueryInterface::getClosestPoints()
		/// if no body information was provided.
		HK_FORCE_INLINE hknpBodyId getQueryBodyId() const;

		/// Returns the shape key of the query shape's leaf shape that is closest to the hit shape.
		HK_FORCE_INLINE hknpShapeKey getQueryShapeKey() const;

		/// Returns the material id of the query shape's leaf shape that is closest to the hit shape.
		HK_FORCE_INLINE hknpMaterialId getQueryShapeMaterialId() const;

		/// Returns the collision filter info of the query shape's leaf shape that is closest to the hit shape.
		HK_FORCE_INLINE hkUint32 getQueryShapeCollisionFilterInfo() const;

		/// Returns the user data of the query shape's leaf shape that is closest to the hit shape.
		HK_FORCE_INLINE hkUint64 getQueryShapeUserData() const;

		/// Returns the hit shape's associated body ID.
		/// Can be HK_NULL for calls to hknpShapeQueryInterface::getClosestPoints() if no body information was provided.
		HK_FORCE_INLINE hknpBodyId getHitBodyId() const;

		/// Returns the shape key of the hit shape's leaf shape that is closest to the query shape.
		HK_FORCE_INLINE hknpShapeKey getHitShapeKey() const;

		/// Returns the material id of the hit shape's leaf shape that is closest to the query shape.
		HK_FORCE_INLINE hknpMaterialId getHitShapeMaterialId() const;

		/// Returns the collision filter info of the hit shape's leaf shape that is closest to the query shape.
		HK_FORCE_INLINE hkUint32 getHitShapeCollisionFilterInfo() const;

		/// Returns the user data of the hit shape's leaf shape that is closest to the query shape.
		HK_FORCE_INLINE hkUint64 getHitShapeUserData() const;
};


/// A collection of accessor methods to an AABB query's result data.
struct hknpAabbQueryResult : public hknpCollisionResult
{
	public:

		/// Returns the body ID associated with the shape that is overlapping the AABB.
		/// Can be HK_NULL for calls to hknpShapeQueryInterface::queryAabb() if no body information was provided.
		HK_FORCE_INLINE hknpBodyId getBodyId() const;

		/// Returns the shape key of the leaf shape that is overlapping the AABB.
		HK_FORCE_INLINE hknpShapeKey getShapeKey() const;

		/// Returns the material id of the leaf shape that is overlapping the AABB.
		HK_FORCE_INLINE hknpMaterialId getShapeMaterialId() const;

		/// Returns the collision filter info of the leaf shape that is overlapping the AABB.
		HK_FORCE_INLINE hkUint32 getShapeCollisionFilterInfo() const;

		/// Returns the user data of the leaf shape that is overlapping the AABB.
		HK_FORCE_INLINE hkUint64 getShapeUserData() const;
};


#include <Physics/Physics/Collide/Query/hknpCollisionResultAccessors.inl>


#endif // HKNP_COLLISION_RESULT_ACCESSORS_H

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
