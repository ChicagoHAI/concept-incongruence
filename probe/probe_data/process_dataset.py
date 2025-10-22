import json

new_dataset = []
with open('death_probe/data/labeled_with_death_year.json', 'r') as f:
    death_data = json.load(f)

for item in death_data:
    if type(item['death_year']) == str and "BY" in item['death_year']:
        continue
    if item['death_year'] == "-1" or item['death_year'] == "30 BC":
        continue
    if type(item['death_year']) == str and int(item['death_year']) > 2025:
        continue
    if item['death_year'] == 2025:
        continue

    new_dataset.append(item)

with open('death_probe/data/labeled_with_death_year_cleaned.json', 'w') as f:
    json.dump(new_dataset, f, indent=4)

# breakpoint()

# # Append the first 250 entries from alive_data to death_data
# death_data.extend(alive_data[-170:])

# # Write the updated death data back to death.json
# with open('death_new.json', 'w') as f:
#     json.dump(death_data, f, indent=4)  # Compact format

# Calculate statistics for the 'isDead' field
is_dead_count = sum(1 for entry in new_dataset if entry.get('isDead') == 1)
is_alive_count = sum(1 for entry in new_dataset if entry.get('isDead') == 0)

print(f"Number of dead: {is_dead_count}")
print(f"Number of alive: {is_alive_count}")

# Calculate statistics for the 'source' field
has_source_count = sum(1 for entry in new_dataset if entry.get('label') == 1)
no_source_count = len(new_dataset) - has_source_count

print(f"Number with source: {has_source_count}")
print(f"Number without source: {no_source_count}")




